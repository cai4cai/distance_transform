// Copyright 2023 Tom Vercauteren. All rights reserved.
//
// This software is licensed under the Apache 2 License.
// See the LICENSE file for details.

// #include <iostream>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "distance_transform/distance_transform.h"

namespace py = pybind11;

template <typename T, int N>
py::array_t<T> DistanceTransformImpl(
    py::array_t<T, py::array::c_style | py::array::forcecast> maskarray,
    bool computeSquareDistance) {
  // Get input shape
  typename dope::DopeVector<T, N>::IndexD masksize, pymasksize;
  std::copy_n(maskarray.shape(), N, pymasksize.begin());
  std::copy_n(maskarray.shape(), N, masksize.begin());

  // Wrap input mask
  // DopeVector would need fixing to avoid the const_cast
  T* ptr = const_cast<T*>(maskarray.data());
  const dope::DopeVector<T, N> dopemask(ptr, 0, masksize);

  // Create output variable
  dope::Grid<T, N> dopefield(masksize);

  dt::DistanceTransform::distanceTransformL2(dopemask, dopefield,
                                             computeSquareDistance);

  return py::array_t<T>(pymasksize, dopefield.data());
}

template <typename T>
py::array_t<T> DistanceTransform(
    py::array_t<T, py::array::c_style | py::array::forcecast> maskarray,
    bool computeSquareDistance = false) {
  // std::cout<<"Got input with dtype="<<maskarray.dtype()<<",
  // ndim="<<maskarray.ndim()<<std::endl;
  switch (maskarray.ndim()) {
    case 1: {
      return DistanceTransformImpl<T, 1>(maskarray, computeSquareDistance);
    }
    case 2: {
      return DistanceTransformImpl<T, 2>(maskarray, computeSquareDistance);
    }
    case 3: {
      return DistanceTransformImpl<T, 3>(maskarray, computeSquareDistance);
    }
    case 4: {
      return DistanceTransformImpl<T, 4>(maskarray, computeSquareDistance);
    }
    case 5: {
      return DistanceTransformImpl<T, 5>(maskarray, computeSquareDistance);
    }
    case 6: {
      return DistanceTransformImpl<T, 6>(maskarray, computeSquareDistance);
    }
    default: {
      throw std::out_of_range("Dimension " + std::to_string(maskarray.ndim()) +
                              " is out of the compiled range");
    }
  }
}

PYBIND11_MODULE(py_distance_transform, m) {
  m.doc() = R"pbdoc(
        Python bindings for the distance transform
        -----------------------
        .. currentmodule:: py_distance_transform
        .. autosummary::
           :toctree: _generate
           distance_transform
    )pbdoc";

  m.def("distance_transform", &DistanceTransform<float>, R"pbdoc(
      Compute the distance transform
      https://github.com/tvercaut/distance_transform
  )pbdoc",
        py::arg("maskarray"), py::arg("computeSquareDistance") = false);
}
