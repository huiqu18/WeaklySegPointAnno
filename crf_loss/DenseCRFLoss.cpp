//
// Created by hui on 11/15/18.
//

#include <torch/torch.h>
#include <iostream>
#include <vector>
using namespace std;
using namespace at;

#include "permutohedral.hpp"


void initializePermutohedral(const float * image, int img_w, int img_h, float sigmargb, float sigmaxy, Permutohedral & lattice_){
    float * features = new float[img_w * img_h * 5];
    for( int j=0; j<img_h; j++ ){
        for( int i=0; i<img_w; i++ ){
            int idx = j*img_w + i;
            features[idx*5+0] = float(i) / sigmaxy;
            features[idx*5+1] = float(j) / sigmaxy;
            features[idx*5+2] = float(image[0*img_w*img_h + idx]) / sigmargb;
            features[idx*5+3] = float(image[1*img_w*img_h + idx]) / sigmargb;
            features[idx*5+4] = float(image[2*img_w*img_h + idx]) / sigmargb;
        }
    }

    lattice_.init( features, 5, img_w * img_h );
    delete [] features;
}

Tensor crfloss_forward(const Tensor input, const Tensor image, float sigma_xy, float sigma_rgb)
{
    IntList size = input.sizes();  // size of input tensor, BxCxHxW expected
    int64_t batch = size[0];    // number of images in a batch
    int64_t height = size[2];    // height of an image
    int64_t width = size[3];    // width of an image

    Tensor losses = torch::autograd::make_variable(at::zeros({batch}));
    Tensor WS = torch::autograd::make_variable(at::zeros({batch, height, width}));

    for( int b=0; b<batch; b++ ){
        const float * image_b = image[b].data<float>();

        // initialize permutohedrals
        Permutohedral lattice_b;
        initializePermutohedral(image_b, width, height, sigma_rgb, sigma_xy, lattice_b);

        auto mask_b = (1 - (image[b][0] < 1e-5) * (image[b][1] < 1e-5) * (image[b][2] < 1e-5)).toType(kFloat);

        // compute DenseCRFloss for current image
        auto prob_map = mask_b * input[b][1].toType(kFloat);
        lattice_b.compute(WS[b].data<float>(), prob_map.data<float>(), 1);
        losses[b] = ((1-input[b][1].toType(kFloat)) * mask_b * WS[b]).sum() / mask_b.sum();   //  SW(1-S)

    }
    auto loss = losses.mean();
    return loss;
}


Tensor crfloss_backward(Tensor grad_output, const Tensor input, const Tensor image, float sigma_xy, float sigma_rgb)
{
    IntList size = input.sizes();  // size of input tensor, Bx2xHxW expected
    int64_t batch = size[0];    // number of images in a batch
    int64_t height = size[2];    // height of an image
    int64_t width = size[3];    // width of an image

    Tensor grad_input = zeros_like(input.toType(kFloat));

    // compute gradient for each input
    for( int b=0; b<batch; b++ ) {
        const float * image_b = image[b].data<float>();

        // initialize permutohedrals
        Permutohedral lattice_b;
        initializePermutohedral(image_b, width, height, sigma_rgb, sigma_xy, lattice_b);

        auto mask_b = (1 - (image[b][0] < 1e-5) * (image[b][1] < 1e-5) * (image[b][2] < 1e-5)).toType(kFloat);
        auto tmp_S = mask_b * (1 - 2 * input[b][1].toType(kFloat));    // W(1-2S)
        lattice_b.compute(grad_input[b][1].data<float>(), tmp_S.data<float>(), 1);
        grad_input[b][1] = grad_input[b][1] * mask_b / mask_b.sum();
        grad_input[b][0] = - grad_input[b][1];
    }

    grad_input = grad_input * grad_output / batch;
    return grad_input.toType(input.type());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &crfloss_forward, "CRFLoss forward");
    m.def("backward", &crfloss_backward, "CRFLoss backward");
}
