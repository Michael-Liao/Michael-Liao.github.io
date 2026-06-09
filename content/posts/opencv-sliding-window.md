+++
date = '2026-06-07T14:18:06+08:00'
draft = false
title = 'OpenCV Sliding Window Processing with C++ Iterators'
tags = ['c++', 'computer-vision']
categories = ['software engineering']
+++
{{< katex >}}

## Background

Sliding window processing is a very common procedure in image processing due to its ability to reveal spatial relationships.
From the most common gaussian/laplacian filters to filter banks such as Gabor and steerable filters, sliding window processing is typically applied to one or more _kernel_.
A kernel, in the context of 2D filters, is a bunch of numbers in the sliding window. Kernels are "applied" to the image via a dot product, where each element of the kernel is multiplied by the elements of the image in the sliding window. Then the sum of all the multiplication is written back into the centre, basically like a convolution.
However, this only covers dot products. What about other algorithms?

In Python, this can be done using [`skimage.filter.rank`](https://scikit-image.org/docs/stable/api/skimage.filters.rank.html), which allows the users to calculate statisical properties in a window.
Unfortunately, there is no equivilent of the `skimage.filter.rank` module in OpenCV.

In this article, I'll show you how to build a generic sliding window processing function that works similarly to the skimage implementation. Or better, to allow _any_ algorithm to be executed in the sliding window fasion.

## Fun with Iterators

C++ iterators is a great tool to abstract away details of iteration, which mainly consists of incrementing to the next position, and retreiving the value.
In our use case, iterators are beneficial to generate `cv::Rect` to iterator over the image.
The following code block shows a minimal working declaration for such an iterator.

```c++
class Sliding2dIterator {
 public:
  using value_type = cv::Rect const;
  using reference = value_type&;
  using pointer = void;
  using different_type = std::ptrdiff_t;
  using iterator_category = std::input_iterator_tag;

  Sliding2dIterator(cv::Size img_size, cv::Size ksize);
  Sliding2dIterator& operator++();
  Sliding2dIterator operator++(int);
  bool operator!=(std::default_sentinel_t) const;
  reference operator*() const;

 private:
  cv::Rect img_extent_;
  cv::Rect roi_;
};
```

The `Sliding2dIterator`'s constructor takes the image size and the kernel size as input arguments.
The kernel size is the size of the window, and the image size is the total bounds that the iterator will stop at.
As an input iterator, it is required to implement both the dereference and the pre/post-increment operators.
Finally, putting some C++20 spice into this article, sentinels will be used to stop the sliding window iterator.

For the iterator's private members, the image extent and the region of interest (ROI) is stored instead of only the size.
This will be crucial for stopping the iterator, as we will shown in a second.

The implementation of `Sliding2dIterator`'s constructor is fairly straightforward.

```c++
Sliding2dIterator::Sliding2dIterator(cv::Size img_size, cv::Size ksize)
    : img_extent_{{0, 0}, img_size}, roi_{{0, 0}, ksize} {}
```

Since the main goal is to iterate over the full image, the private members are initialized using the sizes passed in by arguments with the origin set to  `cv::Point{0, 0}`. Here, I use the power of aggregated initializers to infer the `cv::Point` type and simplify this statement.

The pre-increment operator can be implemented as listed below.
First, it tries to shift the window to the right.
If the right edge of the window exceeds the image width, it resets the x direction and shift the y down by 1.

```c++
Sliding2dIterator& Sliding2dIterator::operator++() {
  // try shifting right
  roi_.x += 1;
  if (roi_.x + roi_.width > img_extent_.width) {
    // next row if exceeded boundary
    roi_.x = 0;
    roi_.y += 1;
  }
  return *this;
}
```

The post-increment operator can be implemented using the pre-increment operator as listed below.
First, a copy of the iterator is created.
Then, the pre-increment operator is called to increment the current iterator.
Finally, the copied iterator is returned.
This ensures the iterator follows the post-increment rules of "return the unchanged and then increment".

```c++
Sliding2dIterator Sliding2dIterator::operator++(int) {
  Sliding2dIterator tmp{*this};
  ++*this;
  return tmp;
}
```

To stop the iterator, every iterator must either compare to itself or to a sentinel type.
There is a cool trick here. `cv::Rect` has a undocumented bitwise and operator (`&`), which will calculate the overlap of the two rects.
This is a simple way to test whether the window has exceed the image boundary.
If the window exceeds the image boundary, the overlapped rect will not be the same as the window.

```c++
bool Sliding2dIterator::operator!=(std::default_sentinel_t) const {
  return (roi_ & img_extent_) == roi_;
}
```

Last but not least, the dereference operator is defined as follows, which simply returns the calculated window.

```c++
cv::Rect const& Sliding2dIterator::operator*() const { return roi_; }
```

In addition to the iterator's class methods, we also define the `begin` and `end` functions _under the same namespace_ as follows.

```c++
Sliding2dIterator begin(Sliding2dIterator it) { return it; }
std::default_sentinel_t end(Sliding2dIterator it) { return std::default_sentinel; }
```

This will allow the `Sliding2dIterator` being used in a ranged for loop, which will simplify our usage further.

## Usage Example

In this section, I will show two examples.
The first one is a non-trivial example implementing `skimage.filters.rank.entropy`.
This example is chosen to show that arbitrary work can be done in the loop.
The second is a trivial example showing an interaction between two images, which is beyond skimage's capabilities.

### Local Entropy

Before getting into the implementation, let's go through a bit of theory.
Entropy, specifically Shannon entropy, measures the amount of bits needed to encode a piece of information.
The equation is defined as follows:

$$
H(X) = -\sum_{x \in X}{p(x)\log_2p(x)}
$$
where \(H(X)\) is the entropy of random variable \(X\); \(p(x)\) is the probability.

In case of images, the closes thing that resembles number of samples is a histogram.
The probability of a grayscale value can be acquired by dividing the histogram with the total population.
The total population is the number of pixels in an image.
However, if a mask is used when collecting the histogram, only the pixels in the mask should be counted in the population.
This provides us with the necessary steps to compute entropy.

1. Build a histogram of the subimage (or in a mask) extracted by a ROI.
2. Divide the histogram with the population and use the eqation above to calculate entropy.

To follow the skimage API, the function must take in an `image` and a `footprint`.
A footprint in opencv is called a kernel.
Our function signature can thus be declared as follows.

```c++
cv::Mat LocalEntropy(cv::Mat img, cv::Mat kernel);
```

The skeleton of the function looks like this.

```c++
cv::Mat entropy_img = cv::Mat::zeros(output_size, CV_64FC1);

int const valid_population = cv::countNonZero(kernel);
cv::Size const output_size = img.size() - (kernel.size() / 2) * 2;
for (auto const& roi : Sliding2dIterator{img.size(), kernel.size()}) {
// ... algorithm to calculate entropy
}
```

`valid_population` is the population in the kernel, which might not be a full rect (e.g. ellipses).
`output_size` is the resulting size of this operation, which is the image size with half of the kernel size trimed on both directions.
It is noted that division by 2 and then times 2 is necessary as we exploit the round down property of integer division.
This allows the half kernel size to be `5` when the kernel size is both `10` and `11`, which will give us the correct padding size of `10` in both even and odd cases.

The following code is used to build the histogram, which is almost the same as the example in OpenCV docs.

```c++
// in for loop body
cv::Mat patch = img(roi);

// histogram parameters
constexpr int channels = 0;
constexpr int n_bins = 256;
constexpr std::array<float, 2> range{0, 256};
float const* ranges = range.data();
cv::Mat hist;
cv::calcHist(&patch, 1, &channels, kernel, hist, 1, &n_bins, &ranges);
```

Then, the histogram is divided by the population to get the probability.

```c++
hist /= valid_population;
```

Finally, the code to compute and update the `entropy_img` is listed as follows.

```c++ {hl_lines=["2-4"]}
hist.forEach<float>([](float& val, int const* pos) {
if (val > 0) {
    val = std::log2(val);
}
});
double entropy = -cv::sum(hist)[0];
entropy_img.at<double>(cv::Point{roi.tl()}) = entropy;
```

The reason why `cv::Mat::forEach` is used instead of `cv::Log` is due to histograms having zeros present.
`cv::Log` will return `-inf` for those values which results in an entropy of `-nan`.
To prevent this, the if statement in lines 3 -- 5 ensure log is only applied to values greater than 0.

### Interaction Between Images

The previous example shows a possible way to compute the local entropy using the `Sliding2dIterator`.
However, it merely mimics the skimage API.
To show the true power of using iterators, I would like to add a basic example with interaction between images.
Suppose you want to compare images patch per patch, and count how many pixels are greater in image1 vs image2.
The implementation would look something like this:

```c++
cv::Mat gt_img = cv::Mat::zeros(output_size, CV_32SC1);

cv::Size const output_size = img.size() - (kernel.size() / 2) * 2;
for (auto const& roi : Sliding2dIterator{img.size(), kernel.size()}) {
    int32_t n_gt_elements = cv::sum(img1(roi) > img2(roi))[0];
    gt_img.at<int32_t>(cv::Point{roi.tl()}) = n_gt_elements;
}
```

As shown above, we can iterate both images in the same loop, making it suitable for more complex algorithms.

## Conclusion

In this article, a C++ iterator is implemented and demonstrated to perform sliding window processing.
The `Sliding2dIterator` simplifies iteration over images by using ranged for loops and sentinels.
It is truly a Zero-Overhead abstraction that improves readability.
Unfortunately, there is an obvious caveat -- Iterating through pixels sequentially takes way too long.
In conclusion, this method is not suitable for real-world use.
Nevertheless, it is a decent starting point to implement generic sliding windows algorithms.

In the next article, I will show how to parallelize this using OpenCV's universal parallel framework (`cv::parallel_for_`).
