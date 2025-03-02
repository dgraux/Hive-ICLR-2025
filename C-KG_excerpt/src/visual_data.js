var final_extraction = 
{
    "Intel/dpt-hybrid-midas": {
        "model_name": "dpt-hybrid-midas",
        "org": "Intel",
        "model_info": {
            "id": "Intel/dpt-hybrid-midas",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 221142,
            "likes": 50,
            "library_name": "transformers",
            "tags": [
                "transformers",
                "pytorch",
                "dpt",
                "depth-estimation",
                "vision",
                "arxiv:2103.13413",
                "license:apache-2.0",
                "model-index",
                "endpoints_compatible",
                "has_space",
                "region:us"
            ],
            "pipeline_tag": "depth-estimation",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "638f07977559bf9a2b2b04ac",
            "createdAt": "2022-12-06T09:12:55.000Z",
            "modelId": "Intel/dpt-hybrid-midas"
        },
        "card_to_dict": {
            "license": "apache-2.0",
            "tags": [
                "vision",
                "depth-estimation"
            ],
            "widget": [
                {
                    "src": "https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg",
                    "example_title": "Tiger"
                },
                {
                    "src": "https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg",
                    "example_title": "Teapot"
                },
                {
                    "src": "https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg",
                    "example_title": "Palace"
                }
            ],
            "model-index": [
                {
                    "name": "dpt-hybrid-midas",
                    "results": [
                        {
                            "task": {
                                "type": "monocular-depth-estimation",
                                "name": "Monocular Depth Estimation"
                            },
                            "dataset": {
                                "name": "MIX-6",
                                "type": "MIX-6"
                            },
                            "metrics": [
                                {
                                    "type": "Zero-shot transfer",
                                    "value": 11.06,
                                    "name": "Zero-shot transfer",
                                    "config": "Zero-shot transfer",
                                    "verified": false
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        "relevant_websites": [
            "https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg",
            "https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg",
            "https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg",
            "https://arxiv.org/abs/2103.13413",
            "https://github.com/isl-org/DPT",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dpt_architecture.jpg",
            "https://huggingface.co/google/vit-hybrid-base-bit-384",
            "https://arxiv.org/abs/2103.13413",
            "https://github.com/isl-org/DPT",
            "https://huggingface.co/Intel/dpt-hybrid-midas/discussions",
            "https://discord.gg/rv2Gp55UJQ",
            "https://huggingface.co/models?search=dpt",
            "https://huggingface.co/docs/transformers/master/en/model_doc/dpt",
            "https://arxiv.org/abs/2103.13413",
            "https://arxiv.org/abs/2103.13413",
            "https://arxiv.org/abs/2103.13413",
            "https://dblp.org/rec/journals/corr/abs-2103-13413.bib",
            "https://dblp.org"
        ],
        "text": "license: apache-2.0 tags: - vision - depth-estimation widget: - src:    example_title: Tiger - src:    example_title: Teapot - src:    example_title: Palace model-index: - name: dpt-hybrid-midas   results:   - task:       type: monocular-depth-estimation       name: Monocular Depth Estimation     dataset:       name: MIX-6       type: MIX-6     metrics:     - type: Zero-shot transfer       value: 11.06       name: Zero-shot transfer       config: Zero-shot transfer       verified: false  Model Details: DPT-Hybrid Dense Prediction Transformer (DPT) model trained on 1.4 million images for monocular depth estimation.  It was introduced in the paper Vision Transformers for Dense Prediction by Ranftl et al. (2021) and first released in this repository.  DPT uses the Vision Transformer (ViT) as backbone and adds a neck + head on top for monocular depth estimation.  This repository hosts the \"hybrid\" version of the model as stated in the paper. DPT-Hybrid diverges from DPT by using ViT-hybrid as a backbone and taking some activations from the backbone. The model card has been written in combination by the Hugging Face team and Intel. How to use Here is how to use this model for zero-shot depth estimation on an image: For more code examples, we refer to the documentation. Quantitative Analyses 12.51 (-12.5%) Table 1. Comparison to the state of the art on monocular depth estimation. We evaluate zero-shot cross-dataset transfer according to the protocol defined in [30]. Relative performance is computed with respect to the original MiDaS model [30]. Lower is better for all metrics. (Ranftl et al., 2021) BibTeX entry and citation info bibtex @article{DBLP:journals/corr/abs-2103-13413,   author    = {Ren{\\'{e}} Ranftl and                Alexey Bochkovskiy and                Vladlen Koltun},   title     = {Vision Transformers for Dense Prediction},   journal   = {CoRR},   volume    = {abs/2103.13413},   year      = {2021},   url       = {},   eprinttype = {arXiv},   eprint    = {2103.13413},   timestamp = {Wed, 07 Apr 2021 15:31:46 +0200},   biburl    = {},   bibsource = {dblp computer science bibliography, } }",
        "markdown_text": "---\nlicense: apache-2.0\ntags:\n- vision\n- depth-estimation\nwidget:\n- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg\n  example_title: Tiger\n- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg\n  example_title: Teapot\n- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg\n  example_title: Palace\nmodel-index:\n- name: dpt-hybrid-midas\n  results:\n  - task:\n      type: monocular-depth-estimation\n      name: Monocular Depth Estimation\n    dataset:\n      name: MIX-6\n      type: MIX-6\n    metrics:\n    - type: Zero-shot transfer\n      value: 11.06\n      name: Zero-shot transfer\n      config: Zero-shot transfer\n      verified: false\n---\n\n## Model Details: DPT-Hybrid \n\nDense Prediction Transformer (DPT) model trained on 1.4 million images for monocular depth estimation. \nIt was introduced in the paper [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413) by Ranftl et al. (2021) and first released in [this repository](https://github.com/isl-org/DPT). \nDPT uses the Vision Transformer (ViT) as backbone and adds a neck + head on top for monocular depth estimation.\n![model image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/dpt_architecture.jpg)\n\nThis repository hosts the \"hybrid\" version of the model as stated in the paper. DPT-Hybrid diverges from DPT by using [ViT-hybrid](https://huggingface.co/google/vit-hybrid-base-bit-384) as a backbone and taking some activations from the backbone.\n\nThe model card has been written in combination by the Hugging Face team and Intel.\n\n| Model Detail | Description |\n| ----------- | ----------- | \n| Model Authors - Company | Intel | \n| Date | December 22, 2022 | \n| Version | 1 | \n| Type | Computer Vision - Monocular Depth Estimation | \n| Paper or Other Resources | [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413) and [GitHub Repo](https://github.com/isl-org/DPT) | \n| License | Apache 2.0 |\n| Questions or Comments | [Community Tab](https://huggingface.co/Intel/dpt-hybrid-midas/discussions) and [Intel Developers Discord](https://discord.gg/rv2Gp55UJQ)|\n\n| Intended Use | Description |\n| ----------- | ----------- | \n| Primary intended uses | You can use the raw model for zero-shot monocular depth estimation. See the [model hub](https://huggingface.co/models?search=dpt) to look for fine-tuned versions on a task that interests you. | \n| Primary intended users | Anyone doing monocular depth estimation | \n| Out-of-scope uses | This model in most cases will need to be fine-tuned for your particular task.  The model should not be used to intentionally create hostile or alienating environments for people.|\n\n### How to use\n\nHere is how to use this model for zero-shot depth estimation on an image:\n\n```python\nfrom PIL import Image\nimport numpy as np\nimport requests\nimport torch\n\nfrom transformers import DPTForDepthEstimation, DPTFeatureExtractor\n\nmodel = DPTForDepthEstimation.from_pretrained(\"Intel/dpt-hybrid-midas\", low_cpu_mem_usage=True)\nfeature_extractor = DPTFeatureExtractor.from_pretrained(\"Intel/dpt-hybrid-midas\")\n\nurl = \"http://images.cocodataset.org/val2017/000000039769.jpg\"\nimage = Image.open(requests.get(url, stream=True).raw)\n\n# prepare image for the model\ninputs = feature_extractor(images=image, return_tensors=\"pt\")\n\nwith torch.no_grad():\n    outputs = model(**inputs)\n    predicted_depth = outputs.predicted_depth\n\n# interpolate to original size\nprediction = torch.nn.functional.interpolate(\n    predicted_depth.unsqueeze(1),\n    size=image.size[::-1],\n    mode=\"bicubic\",\n    align_corners=False,\n)\n\n# visualize the prediction\noutput = prediction.squeeze().cpu().numpy()\nformatted = (output * 255 / np.max(output)).astype(\"uint8\")\ndepth = Image.fromarray(formatted)\ndepth.show()\n```\n\nFor more code examples, we refer to the [documentation](https://huggingface.co/docs/transformers/master/en/model_doc/dpt).\n\n| Factors | Description | \n| ----------- | ----------- | \n| Groups | Multiple datasets compiled together | \n| Instrumentation | - |\n| Environment | Inference completed on Intel Xeon Platinum 8280 CPU @ 2.70GHz with 8 physical cores and an NVIDIA RTX 2080 GPU. |\n| Card Prompts | Model deployment on alternate hardware and software will change model performance |\n\n| Metrics | Description | \n| ----------- | ----------- | \n| Model performance measures | Zero-shot Transfer |\n| Decision thresholds | - | \n| Approaches to uncertainty and variability | - |\n\n| Training and Evaluation Data | Description | \n| ----------- | ----------- | \n| Datasets | The dataset is called MIX 6, and contains around 1.4M images. The model was initialized with ImageNet-pretrained weights.|\n| Motivation | To build a robust monocular depth prediction network |\n| Preprocessing | \"We resize the image such that the longer side is 384 pixels and train on random square crops of size 384. ... We perform random horizontal flips for data augmentation.\" See [Ranftl et al. (2021)](https://arxiv.org/abs/2103.13413) for more details. | \n\n## Quantitative Analyses\n| Model | Training set | DIW WHDR | ETH3D AbsRel | Sintel AbsRel | KITTI δ>1.25 | NYU δ>1.25 | TUM δ>1.25 |\n| --- | --- | --- | --- | --- | --- | --- | --- | \n| DPT - Large | MIX 6 | 10.82 (-13.2%) | 0.089 (-31.2%) | 0.270 (-17.5%) | 8.46 (-64.6%) | 8.32 (-12.9%) | 9.97 (-30.3%) |\n| DPT - Hybrid | MIX 6 | 11.06 (-11.2%) | 0.093 (-27.6%) | 0.274 (-16.2%) | 11.56 (-51.6%) | 8.69 (-9.0%) | 10.89 (-23.2%) | \n| MiDaS  | MIX 6  | 12.95 (+3.9%)  | 0.116 (-10.5%)  | 0.329 (+0.5%)  | 16.08 (-32.7%)  | 8.71 (-8.8%)  | 12.51 (-12.5%)\n| MiDaS [30]  | MIX 5  | 12.46  | 0.129  | 0.327  | 23.90  | 9.55  | 14.29 | \n | Li [22]  | MD [22]  | 23.15  | 0.181  | 0.385  | 36.29  | 27.52  | 29.54 | \n | Li [21]  | MC [21]  | 26.52  | 0.183  | 0.405  | 47.94  | 18.57  | 17.71 | \n | Wang [40]  | WS [40]  | 19.09  | 0.205  | 0.390  | 31.92  | 29.57  | 20.18 | \n | Xian [45]  | RW [45]  | 14.59  | 0.186 |  0.422  | 34.08 |  27.00 |  25.02 | \n | Casser [5]  | CS [8]  | 32.80  | 0.235  | 0.422  | 21.15  | 39.58  | 37.18 | \n\nTable 1. Comparison to the state of the art on monocular depth estimation. We evaluate zero-shot cross-dataset transfer according to the\nprotocol defined in [30]. Relative performance is computed with respect to the original MiDaS model [30]. Lower is better for all metrics. ([Ranftl et al., 2021](https://arxiv.org/abs/2103.13413))\n\n\n| Ethical Considerations | Description | \n| ----------- | ----------- | \n| Data | The training data come from multiple image datasets compiled together. |\n| Human life | The model is not intended to inform decisions central to human life or flourishing. It is an aggregated set of monocular depth image datasets. | \n| Mitigations | No additional risk mitigation strategies were considered during model development. |\n| Risks and harms | The extent of the risks involved by using the model remain unknown. |\n| Use cases | - | \n\n| Caveats and Recommendations |\n| ----------- | \n| Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. There are no additional caveats or recommendations for this model. |\n\n### BibTeX entry and citation info\n\n```bibtex\n@article{DBLP:journals/corr/abs-2103-13413,\n  author    = {Ren{\\'{e}} Ranftl and\n               Alexey Bochkovskiy and\n               Vladlen Koltun},\n  title     = {Vision Transformers for Dense Prediction},\n  journal   = {CoRR},\n  volume    = {abs/2103.13413},\n  year      = {2021},\n  url       = {https://arxiv.org/abs/2103.13413},\n  eprinttype = {arXiv},\n  eprint    = {2103.13413},\n  timestamp = {Wed, 07 Apr 2021 15:31:46 +0200},\n  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-13413.bib},\n  bibsource = {dblp computer science bibliography, https://dblp.org}\n}\n```",
        "llm_extraction": {
            "model_name": "DPT-Hybrid Dense Prediction Transformer (DPT)",
            "model_framework": "NONE",
            "model_architecture": "Vision Transformer",
            "tasks": [
                "monocular-depth-estimation"
            ],
            "training_strategy": "monocular depth estimation",
            "parameters": "1.4 million images",
            "vocab_size": "NONE",
            "training_data": "MIX-6 dataset",
            "authors": [
                "René Ranftl",
                "Alexey Bochkovskiy",
                "Vladlen Koltun"
            ],
            "other": [
                "Zero-shot transfer",
                "Vision Transformer",
                "ViT-hybrid",
                "bibtex",
                "citation info"
            ]
        },
        "truncation": 0,
        "extraction_version": "v_1",
        "papers_with_code": "successful",
        "performance": [
            {
                "id": "e9eb04a7-1806-4228-907c-64488e93c8ab",
                "best_rank": 1.0,
                "metrics": {
                    "Validation mIoU": "49.02"
                },
                "methodology": "DPT-Hybrid",
                "uses_additional_data": false,
                "paper": "vision-transformers-for-dense-prediction",
                "best_metric": "Validation mIoU",
                "evaluated_on": "2021-03-24",
                "evaluation": "semantic-segmentation-on-ade20k",
                "benchmark_details": {
                    "id": "semantic-segmentation-on-ade20k",
                    "task": "semantic-segmentation",
                    "dataset": "ade20k",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "199b4792-69a2-4b88-b5c9-5cca336ea695",
                "best_rank": NaN,
                "metrics": {
                    "mIoU": "49.02",
                    "Pixel Accuracy": "83.11"
                },
                "methodology": "DPT-Hybrid",
                "uses_additional_data": false,
                "paper": "vision-transformers-for-dense-prediction",
                "best_metric": null,
                "evaluated_on": "2021-03-24",
                "evaluation": "semantic-segmentation-on-ade20k-val",
                "benchmark_details": {
                    "id": "semantic-segmentation-on-ade20k-val",
                    "task": "semantic-segmentation",
                    "dataset": "ade20k-val",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "155976cb-9118-4a07-a93f-2e2acfe76b16",
                "best_rank": NaN,
                "metrics": {
                    "absolute relative error": "0.062",
                    "RMSE": "2.573",
                    "RMSE log": "0.092",
                    "Delta < 1.25": "0.959",
                    "Delta < 1.25^2": "0.995",
                    "Delta < 1.25^3": "0.999"
                },
                "methodology": "DPT-Hybrid",
                "uses_additional_data": false,
                "paper": "vision-transformers-for-dense-prediction",
                "best_metric": null,
                "evaluated_on": "2021-03-24",
                "evaluation": "monocular-depth-estimation-on-kitti-eigen",
                "benchmark_details": {
                    "id": "monocular-depth-estimation-on-kitti-eigen",
                    "task": "monocular-depth-estimation",
                    "dataset": "kitti-eigen-split",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "2c51fec2-0d1d-439e-8e60-3690b7db8d7e",
                "best_rank": 3.0,
                "metrics": {
                    "RMSE": "0.357",
                    "absolute relative error": "0.110",
                    "Delta < 1.25": "0.904",
                    "Delta < 1.25^2": "0.988",
                    "Delta < 1.25^3": "0.994",
                    "log 10": "0.045"
                },
                "methodology": "DPT-Hybrid",
                "uses_additional_data": true,
                "paper": "vision-transformers-for-dense-prediction",
                "best_metric": "RMSE",
                "evaluated_on": "2021-03-24",
                "evaluation": "monocular-depth-estimation-on-nyu-depth-v2",
                "benchmark_details": {
                    "id": "monocular-depth-estimation-on-nyu-depth-v2",
                    "task": "monocular-depth-estimation",
                    "dataset": "nyu-depth-v2-1",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "ad686cdc-7503-4856-8bc9-f4092c8705e2",
                "best_rank": 12.0,
                "metrics": {
                    "mIoU": "60.46"
                },
                "methodology": "DPT-Hybrid",
                "uses_additional_data": false,
                "paper": "vision-transformers-for-dense-prediction",
                "best_metric": "mIoU",
                "evaluated_on": "2021-03-24",
                "evaluation": "semantic-segmentation-on-pascal-context",
                "benchmark_details": {
                    "id": "semantic-segmentation-on-pascal-context",
                    "task": "semantic-segmentation",
                    "dataset": "pascal-context",
                    "description": "",
                    "mirror_url": null
                }
            }
        ],
        "model_usage": {
            "llm_input": {
                "Quantitative Analyses/ BibTeX entry and citation info": "@article{DBLP:journals/corr/abs-2103-13413,\n  author    = {Ren{\\'{e}} Ranftl and\n               Alexey Bochkovskiy and\n               Vladlen Koltun},\n  title     = {Vision Transformers for Dense Prediction},\n  journal   = {CoRR},\n  volume    = {abs/2103.13413},\n  year      = {2021},\n  url       = {https://arxiv.org/abs/2103.13413},\n  eprinttype = {arXiv},\n  eprint    = {2103.13413},\n  timestamp = {Wed, 07 Apr 2021 15:31:46 +0200},\n  biburl    = {https://dblp.org/rec/journals/corr/abs-2103-13413.bib},\n  bibsource = {dblp computer science bibliography, https://dblp.org}\n}"
            },
            "usage": {
                "Model Details: DPT-Hybrid / How to use": "Here is how to use this model for zero-shot depth estimation on an image:\n```\nfrom PIL import Image\nimport numpy as np\nimport requests\nimport torch\nfrom transformers import DPTForDepthEstimation, DPTFeatureExtractor\nmodel = DPTForDepthEstimation.from_pretrained(\"Intel/dpt-hybrid-midas\", low_cpu_mem_usage=True)\nfeature_extractor = DPTFeatureExtractor.from_pretrained(\"Intel/dpt-hybrid-midas\")\nurl = \"http://images.cocodataset.org/val2017/000000039769.jpg\"\nimage = Image.open(requests.get(url, stream=True).raw)\nprepare image for the model\ninputs = feature_extractor(images=image, return_tensors=\"pt\")\nwith torch.no_grad():\n    outputs = model(**inputs)\n    predicted_depth = outputs.predicted_depth\ninterpolate to original size\nprediction = torch.nn.functional.interpolate(\n    predicted_depth.unsqueeze(1),\n    size=image.size[::-1],\n    mode=\"bicubic\",\n    align_corners=False,\n)\nvisualize the prediction\noutput = prediction.squeeze().cpu().numpy()\nformatted = (output * 255 / np.max(output)).astype(\"uint8\")\ndepth = Image.fromarray(formatted)\ndepth.show()\n```\nFor more code examples, we refer to the documentation.\n| Factors | Description | \n| ----------- | ----------- | \n| Groups | Multiple datasets compiled together | \n| Metrics | Description | \n| ----------- | ----------- | \n| Model performance measures | Zero-shot Transfer |\n| Decision thresholds | - | \n| Approaches to uncertainty and variability | - |\n| Training and Evaluation Data | Description | \n| ----------- | ----------- | \n| Preprocessing | \"We resize the image such that the longer side is 384 pixels and train on random square crops of size 384. ... We perform random horizontal flips for data augmentation.\" See Ranftl et al. (2021) for more details. | "
            },
            "model_function": [
                {
                    "code": "from PIL import Image\nimport numpy as np\nimport requests\nimport torch\nfrom transformers import DPTForDepthEstimation, DPTFeatureExtractor\n\ndef estimate_depth(url=\"http://images.cocodataset.org/val2017/000000039769.jpg\",\n                   model_name=\"Intel/dpt-hybrid-midas\"):\n    model = DPTForDepthEstimation.from_pretrained(model_name, low_cpu_mem_usage=True)\n    feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)\n    image = Image.open(requests.get(url, stream=True).raw)\n\n    # prepare image for the model\n    inputs = feature_extractor(images=image, return_tensors=\"pt\")\n\n    with torch.no_grad():\n        outputs = model(inputs)\n        predicted_depth = outputs.predicted_depth\n\n    # interpolate to original size\n    prediction = torch.nn.functional.interpolate(\n        predicted_depth.unsqueeze(1),\n        size=image.size[::-1],\n        mode=\"bicubic\",\n        align_corners=False,\n    )\n\n    # visualize the prediction\n    output = prediction.squeeze().cpu().numpy()\n    formatted = (output * 255 / np.max(output)).astype(\"uint8\")\n    depth = Image.fromarray(formatted)\n    depth.show()",
                    "function_info": {
                        "return": null,
                        "function_name": "estimate_depth",
                        "variables": [
                            {
                                "name": "input",
                                "type": "str",
                                "default": "./elephant.jpeg"
                            }
                        ]
                    }
                }
            ]
        }
    },
    "Salesforce/blip-image-captioning-base": {
        "model_name": "blip-image-captioning-base",
        "org": "Salesforce",
        "model_info": {
            "id": "Salesforce/blip-image-captioning-base",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 748853,
            "likes": 330,
            "library_name": "transformers",
            "tags": [
                "transformers",
                "pytorch",
                "tf",
                "blip",
                "text2text-generation",
                "image-captioning",
                "image-to-text",
                "arxiv:2201.12086",
                "license:bsd-3-clause",
                "autotrain_compatible",
                "endpoints_compatible",
                "has_space",
                "region:us"
            ],
            "pipeline_tag": "image-to-text",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "63974666f74f697677e32689",
            "createdAt": "2022-12-12T15:19:02.000Z",
            "modelId": "Salesforce/blip-image-captioning-base"
        },
        "card_to_dict": {
            "license": "bsd-3-clause",
            "tags": [
                "image-captioning"
            ],
            "pipeline_tag": "image-to-text",
            "languages": [
                "en"
            ]
        },
        "relevant_websites": [
            "https://cdn-uploads.huggingface.co/production/uploads/1670928184033-62441d1d9fdefb55a0b7d12c.gif",
            "https://github.com/salesforce/BLIP",
            "https://arxiv.org/abs/2201.12086",
            "https://doi.org/10.48550/arxiv.2201.12086",
            "https://arxiv.org/abs/2201.12086"
        ],
        "text": "license: bsd-3-clause tags: - image-captioning pipeline_tag: image-to-text languages: - en  BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation Model card for image captioning pretrained on COCO dataset - base architecture (with ViT base backbone). TL;DR Authors from the paper write in the abstract: Vision-Language Pre-training (VLP) has advanced the performance for many vision-language tasks. However, most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision. In this paper, we propose BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. We achieve state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval (+2.7% in average recall@1), image captioning (+2.8% in CIDEr), and VQA (+1.6% in VQA score). BLIP also demonstrates strong generalization ability when directly transferred to videolanguage tasks in a zero-shot manner. Code, models, and datasets are released. Usage You can use this model for conditional and un-conditional image captioning Using the Pytorch model Running the model on CPU   Click to expand   Running the model on GPU In full precision   Click to expand   In half precision (float16)   Click to expand   BibTex and citation info ``` @misc{,   doi = {10.48550/ARXIV.2201.12086}, url = {}, author = {Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven}, keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences}, title = {BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation}, publisher = {arXiv}, year = {2022}, copyright = {Creative Commons Attribution 4.0 International} } ```",
        "markdown_text": "---\nlicense: bsd-3-clause\ntags:\n- image-captioning\npipeline_tag: image-to-text\nlanguages:\n- en\n---\n\n# BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation\n\nModel card for image captioning pretrained on COCO dataset - base architecture (with ViT base backbone).\n\n| ![BLIP.gif](https://cdn-uploads.huggingface.co/production/uploads/1670928184033-62441d1d9fdefb55a0b7d12c.gif) |\n|:--:|\n| <b> Pull figure from BLIP official repo | Image source: https://github.com/salesforce/BLIP </b>|\n\n## TL;DR\n\nAuthors from the [paper](https://arxiv.org/abs/2201.12086) write in the abstract:\n\n*Vision-Language Pre-training (VLP) has advanced the performance for many vision-language tasks. However, most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision. In this paper, we propose BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. We achieve state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval (+2.7% in average recall@1), image captioning (+2.8% in CIDEr), and VQA (+1.6% in VQA score). BLIP also demonstrates strong generalization ability when directly transferred to videolanguage tasks in a zero-shot manner. Code, models, and datasets are released.*\n\n## Usage\n\nYou can use this model for conditional and un-conditional image captioning\n\n### Using the Pytorch model\n\n#### Running the model on CPU\n\n<details>\n<summary> Click to expand </summary>\n\n```python\nimport requests\nfrom PIL import Image\nfrom transformers import BlipProcessor, BlipForConditionalGeneration\n\nprocessor = BlipProcessor.from_pretrained(\"Salesforce/blip-image-captioning-base\")\nmodel = BlipForConditionalGeneration.from_pretrained(\"Salesforce/blip-image-captioning-base\")\n\nimg_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' \nraw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')\n\n# conditional image captioning\ntext = \"a photography of\"\ninputs = processor(raw_image, text, return_tensors=\"pt\")\n\nout = model.generate(**inputs)\nprint(processor.decode(out[0], skip_special_tokens=True))\n# >>> a photography of a woman and her dog\n\n# unconditional image captioning\ninputs = processor(raw_image, return_tensors=\"pt\")\n\nout = model.generate(**inputs)\nprint(processor.decode(out[0], skip_special_tokens=True))\n>>> a woman sitting on the beach with her dog\n```\n</details>\n\n#### Running the model on GPU\n\n##### In full precision \n\n<details>\n<summary> Click to expand </summary>\n\n```python\nimport requests\nfrom PIL import Image\nfrom transformers import BlipProcessor, BlipForConditionalGeneration\n\nprocessor = BlipProcessor.from_pretrained(\"Salesforce/blip-image-captioning-base\")\nmodel = BlipForConditionalGeneration.from_pretrained(\"Salesforce/blip-image-captioning-base\").to(\"cuda\")\n\nimg_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' \nraw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')\n\n# conditional image captioning\ntext = \"a photography of\"\ninputs = processor(raw_image, text, return_tensors=\"pt\").to(\"cuda\")\n\nout = model.generate(**inputs)\nprint(processor.decode(out[0], skip_special_tokens=True))\n# >>> a photography of a woman and her dog\n\n# unconditional image captioning\ninputs = processor(raw_image, return_tensors=\"pt\").to(\"cuda\")\n\nout = model.generate(**inputs)\nprint(processor.decode(out[0], skip_special_tokens=True))\n>>> a woman sitting on the beach with her dog\n```\n</details>\n\n##### In half precision (`float16`)\n\n<details>\n<summary> Click to expand </summary>\n\n```python\nimport torch\nimport requests\nfrom PIL import Image\nfrom transformers import BlipProcessor, BlipForConditionalGeneration\n\nprocessor = BlipProcessor.from_pretrained(\"Salesforce/blip-image-captioning-base\")\nmodel = BlipForConditionalGeneration.from_pretrained(\"Salesforce/blip-image-captioning-base\", torch_dtype=torch.float16).to(\"cuda\")\n\nimg_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' \nraw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')\n\n# conditional image captioning\ntext = \"a photography of\"\ninputs = processor(raw_image, text, return_tensors=\"pt\").to(\"cuda\", torch.float16)\n\nout = model.generate(**inputs)\nprint(processor.decode(out[0], skip_special_tokens=True))\n# >>> a photography of a woman and her dog\n\n# unconditional image captioning\ninputs = processor(raw_image, return_tensors=\"pt\").to(\"cuda\", torch.float16)\n\nout = model.generate(**inputs)\nprint(processor.decode(out[0], skip_special_tokens=True))\n>>> a woman sitting on the beach with her dog\n```\n</details>\n\n## BibTex and citation info\n\n```\n@misc{https://doi.org/10.48550/arxiv.2201.12086,\n  doi = {10.48550/ARXIV.2201.12086},\n  \n  url = {https://arxiv.org/abs/2201.12086},\n  \n  author = {Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},\n  \n  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},\n  \n  title = {BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},\n  \n  publisher = {arXiv},\n  \n  year = {2022},\n  \n  copyright = {Creative Commons Attribution 4.0 International}\n}\n```\n",
        "llm_extraction": {
            "model_name": "BLIP",
            "model_framework": "transformers",
            "model_architecture": "ViT base backbone",
            "tasks": [
                "image-captioning",
                "image-text retrieval",
                "VQA"
            ],
            "training_strategy": "Bootstrapping Language-Image Pre-training",
            "parameters": "NONE",
            "vocab_size": "NONE",
            "data": "COCO dataset",
            "authors": [
                "Li, Junnan",
                "Li, Dongxu",
                "Xiong, Caiming",
                "Hoi, Steven"
            ],
            "other": [
                "vision-language understanding",
                "vision-language generation",
                "generalization ability",
                "PyTorch model",
                "full precision",
                "half precision (float16)",
                "BibTex and citation info"
            ]
        },
        "truncation": 0,
        "extraction_version": "v_1",
        "papers_with_code": "successful",
        "performance": [
            {
                "id": "918bac95-8a78-431c-b24c-a9daf400b53f",
                "best_rank": NaN,
                "metrics": {
                    "ADD(S) AUC": "83.51"
                },
                "methodology": "BLIP",
                "uses_additional_data": false,
                "paper": "blip-bootstrapping-language-image-pre",
                "best_metric": null,
                "evaluated_on": "2022-01-28",
                "evaluation": "image-text-matching-on-commercialadsdataset",
                "benchmark_details": {
                    "id": "image-text-matching-on-commercialadsdataset",
                    "task": "image-text-matching",
                    "dataset": "commercialadsdataset",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "44e9740e-68f9-47e6-86ab-274b20dabfa9",
                "best_rank": 3.0,
                "metrics": {
                    "mean average precision": "24.3"
                },
                "methodology": "BLIP",
                "uses_additional_data": true,
                "paper": "blip-bootstrapping-language-image-pre",
                "best_metric": "mean average precision",
                "evaluated_on": "2022-01-28",
                "evaluation": "open-vocabulary-attribute-detection-on-ovad-1",
                "benchmark_details": {
                    "id": "open-vocabulary-attribute-detection-on-ovad-1",
                    "task": "open-vocabulary-attribute-detection",
                    "dataset": "ovad-box-benchmark",
                    "description": "",
                    "mirror_url": null
                }
            }
        ],
        "model_usage": {
            "llm_input": {
                "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation": {
                    "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation/ TL;DR": "Authors from the paper write in the abstract:\nVision-Language Pre-training (VLP) has advanced the performance for many vision-language tasks. However, most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision. In this paper, we propose BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. We achieve state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval (+2.7% in average recall@1), image captioning (+2.8% in CIDEr), and VQA (+1.6% in VQA score). BLIP also demonstrates strong generalization ability when directly transferred to videolanguage tasks in a zero-shot manner. Code, models, and datasets are released.",
                    "Using the Pytorch model/ Running the model on CPU": "\n Click to expand \n",
                    "Running the model on GPU/ In full precision": "\n Click to expand \n",
                    "Running the model on GPU/ In half precision (`float16`)": "\n Click to expand \n",
                    "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation/ BibTex and citation info": "```\n@misc{https://doi.org/10.48550/arxiv.2201.12086,\n  doi = {10.48550/ARXIV.2201.12086},\nurl = {https://arxiv.org/abs/2201.12086},\nauthor = {Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},\nkeywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},\ntitle = {BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},\npublisher = {arXiv},\nyear = {2022},\ncopyright = {Creative Commons Attribution 4.0 International}\n}\n```"
                }
            },
            "usage": {},
            "model_function": [
                {
                    "function_info": {
                        "return": "str",
                        "function_name": "caption_image",
                        "variables": [
                            {
                                "name": "input",
                                "type": "str",
                                "default": "./elephant.jpeg"
                            }
                        ]
                    }
                }
            ]
        }
    },
    "Salesforce/blip-vqa-base": {
        "model_name": "blip-vqa-base",
        "org": "Salesforce",
        "model_info": {
            "id": "Salesforce/blip-vqa-base",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 643747,
            "likes": 77,
            "library_name": "transformers",
            "tags": [
                "transformers",
                "pytorch",
                "tf",
                "safetensors",
                "blip",
                "question-answering",
                "visual-question-answering",
                "arxiv:2201.12086",
                "license:bsd-3-clause",
                "has_space",
                "region:us"
            ],
            "pipeline_tag": "visual-question-answering",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "63976a3965d1f3432248c26c",
            "createdAt": "2022-12-12T17:51:53.000Z",
            "modelId": "Salesforce/blip-vqa-base"
        },
        "card_to_dict": {
            "license": "bsd-3-clause",
            "tags": [
                "visual-question-answering"
            ],
            "pipeline_tag": "visual-question-answering",
            "inference": false,
            "languages": [
                "en"
            ]
        },
        "relevant_websites": [
            "https://cdn-uploads.huggingface.co/production/uploads/1670928184033-62441d1d9fdefb55a0b7d12c.gif",
            "https://github.com/salesforce/BLIP",
            "https://arxiv.org/abs/2201.12086",
            "https://doi.org/10.48550/arxiv.2201.12086",
            "https://arxiv.org/abs/2201.12086"
        ],
        "text": "license: bsd-3-clause tags: - visual-question-answering pipeline_tag: visual-question-answering inference: false languages: - en  BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation Model card for BLIP trained on visual question answering- base architecture (with ViT base backbone). TL;DR Authors from the paper write in the abstract: Vision-Language Pre-training (VLP) has advanced the performance for many vision-language tasks. However, most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision. In this paper, we propose BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. We achieve state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval (+2.7% in average recall@1), image captioning (+2.8% in CIDEr), and VQA (+1.6% in VQA score). BLIP also demonstrates strong generalization ability when directly transferred to videolanguage tasks in a zero-shot manner. Code, models, and datasets are released. Usage You can use this model for conditional and un-conditional image captioning Using the Pytorch model Running the model on CPU   Click to expand   Running the model on GPU In full precision   Click to expand   In half precision (float16)   Click to expand   BibTex and citation info ``` @misc{,   doi = {10.48550/ARXIV.2201.12086}, url = {}, author = {Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven}, keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences}, title = {BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation}, publisher = {arXiv}, year = {2022}, copyright = {Creative Commons Attribution 4.0 International} } ```",
        "markdown_text": "---\nlicense: bsd-3-clause\ntags:\n- visual-question-answering\npipeline_tag: visual-question-answering\ninference: false\nlanguages:\n- en\n---\n\n# BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation\n\nModel card for BLIP trained on visual question answering- base architecture (with ViT base backbone).\n\n| ![BLIP.gif](https://cdn-uploads.huggingface.co/production/uploads/1670928184033-62441d1d9fdefb55a0b7d12c.gif) |\n|:--:|\n| <b> Pull figure from BLIP official repo | Image source: https://github.com/salesforce/BLIP </b>|\n\n## TL;DR\n\nAuthors from the [paper](https://arxiv.org/abs/2201.12086) write in the abstract:\n\n*Vision-Language Pre-training (VLP) has advanced the performance for many vision-language tasks. However, most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision. In this paper, we propose BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. We achieve state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval (+2.7% in average recall@1), image captioning (+2.8% in CIDEr), and VQA (+1.6% in VQA score). BLIP also demonstrates strong generalization ability when directly transferred to videolanguage tasks in a zero-shot manner. Code, models, and datasets are released.*\n\n## Usage\n\nYou can use this model for conditional and un-conditional image captioning\n\n### Using the Pytorch model\n\n#### Running the model on CPU\n\n<details>\n<summary> Click to expand </summary>\n\n```python\nimport requests\nfrom PIL import Image\nfrom transformers import BlipProcessor, BlipForQuestionAnswering\n\nprocessor = BlipProcessor.from_pretrained(\"Salesforce/blip-vqa-base\")\nmodel = BlipForQuestionAnswering.from_pretrained(\"Salesforce/blip-vqa-base\")\n\nimg_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' \nraw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')\n\nquestion = \"how many dogs are in the picture?\"\ninputs = processor(raw_image, question, return_tensors=\"pt\")\n\nout = model.generate(**inputs)\nprint(processor.decode(out[0], skip_special_tokens=True))\n>>> 1\n```\n</details>\n\n#### Running the model on GPU\n\n##### In full precision \n\n<details>\n<summary> Click to expand </summary>\n\n```python\nimport requests\nfrom PIL import Image\nfrom transformers import BlipProcessor, BlipForQuestionAnswering\n\nprocessor = BlipProcessor.from_pretrained(\"Salesforce/blip-vqa-base\")\nmodel = BlipForQuestionAnswering.from_pretrained(\"Salesforce/blip-vqa-base\").to(\"cuda\")\n\nimg_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' \nraw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')\n\nquestion = \"how many dogs are in the picture?\"\ninputs = processor(raw_image, question, return_tensors=\"pt\").to(\"cuda\")\n\nout = model.generate(**inputs)\nprint(processor.decode(out[0], skip_special_tokens=True))\n>>> 1\n```\n</details>\n\n##### In half precision (`float16`)\n\n<details>\n<summary> Click to expand </summary>\n\n```python\nimport torch\nimport requests\nfrom PIL import Image\nfrom transformers import BlipProcessor, BlipForQuestionAnswering\n\nprocessor = BlipProcessor.from_pretrained(\"ybelkada/blip-vqa-base\")\nmodel = BlipForQuestionAnswering.from_pretrained(\"ybelkada/blip-vqa-base\", torch_dtype=torch.float16).to(\"cuda\")\n\nimg_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' \nraw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')\n\nquestion = \"how many dogs are in the picture?\"\ninputs = processor(raw_image, question, return_tensors=\"pt\").to(\"cuda\", torch.float16)\n\nout = model.generate(**inputs)\nprint(processor.decode(out[0], skip_special_tokens=True))\n>>> 1\n```\n</details>\n\n## BibTex and citation info\n\n```\n@misc{https://doi.org/10.48550/arxiv.2201.12086,\n  doi = {10.48550/ARXIV.2201.12086},\n  \n  url = {https://arxiv.org/abs/2201.12086},\n  \n  author = {Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},\n  \n  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},\n  \n  title = {BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},\n  \n  publisher = {arXiv},\n  \n  year = {2022},\n  \n  copyright = {Creative Commons Attribution 4.0 International}\n}\n```",
        "llm_extraction": "           {\"model_name\": \"BLIP\", \"model_framework\": \"transformers\", \"model_architecture\": \"ViT base backbone\", \"tasks\": [\"image-text retrieval\", \"image captioning\", \"VQA\"], \"training_strategy\": \"Bootstrapping Language-Image Pre-training\", \"parameters\": \"NONE\", \"vocab_size\": \"NONE\", \"data\": \"noisy web data\", \"authors\": [\"Li, Junnan\", \"Li, Dongxu\", \"Xiong, Caiming\", \"Hoi, Steven\"], \"other\": [\"visual-question-answering\", \"pipeline_tag\", \"inference\", \"languages\", \"en\", \"BLIP\", \"Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation Model\", \"card for BLIP trained on visual question answering- base architecture (with ViT base backbone)\", \"TL;DR\", \"Vision-Language Pre-training (VLP)\", \"has advanced the performance for many vision-language tasks\", \"most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks\", \"performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web\", \"which is a suboptimal source of supervision\", \"In this paper, we propose BLIP\", \"a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks\", \"BLIP effectively utilizes the noisy web data by bootstrapping the captions\", \"where a captioner generates synthetic captions and a filter removes the noisy ones\", \"We achieve state-of-the-art results on a wide range of vision-language tasks\", \"such as image-text retrieval (+2.7% in average recall@1)\", \"image captioning (+2.8% in CIDEr)\", and \"VQA (+1.6% in VQA score)\", \"BLIP also demonstrates strong generalization ability when directly transferred to videolanguage tasks in a zero-shot manner\", \"Code, models, and datasets are released\", \"Usage\", \"You can use this model for conditional and un-conditional image captioning\", \"Running the model on CPU",
        "truncation": 0,
        "extraction_version": "v_1",
        "papers_with_code": "successful",
        "performance": [
            {
                "id": "918bac95-8a78-431c-b24c-a9daf400b53f",
                "best_rank": NaN,
                "metrics": {
                    "ADD(S) AUC": "83.51"
                },
                "methodology": "BLIP",
                "uses_additional_data": false,
                "paper": "blip-bootstrapping-language-image-pre",
                "best_metric": null,
                "evaluated_on": "2022-01-28",
                "evaluation": "image-text-matching-on-commercialadsdataset",
                "benchmark_details": {
                    "id": "image-text-matching-on-commercialadsdataset",
                    "task": "image-text-matching",
                    "dataset": "commercialadsdataset",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "44e9740e-68f9-47e6-86ab-274b20dabfa9",
                "best_rank": 3.0,
                "metrics": {
                    "mean average precision": "24.3"
                },
                "methodology": "BLIP",
                "uses_additional_data": true,
                "paper": "blip-bootstrapping-language-image-pre",
                "best_metric": "mean average precision",
                "evaluated_on": "2022-01-28",
                "evaluation": "open-vocabulary-attribute-detection-on-ovad-1",
                "benchmark_details": {
                    "id": "open-vocabulary-attribute-detection-on-ovad-1",
                    "task": "open-vocabulary-attribute-detection",
                    "dataset": "ovad-box-benchmark",
                    "description": "",
                    "mirror_url": null
                }
            }
        ],
        "model_usage": {
            "llm_input": {
                "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation": {
                    "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation/ TL;DR": "Authors from the paper write in the abstract:\nVision-Language Pre-training (VLP) has advanced the performance for many vision-language tasks. However, most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision. In this paper, we propose BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. We achieve state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval (+2.7% in average recall@1), image captioning (+2.8% in CIDEr), and VQA (+1.6% in VQA score). BLIP also demonstrates strong generalization ability when directly transferred to videolanguage tasks in a zero-shot manner. Code, models, and datasets are released.",
                    "Using the Pytorch model/ Running the model on CPU": "\n Click to expand \n",
                    "Running the model on GPU/ In full precision": "\n Click to expand \n",
                    "Running the model on GPU/ In half precision (`float16`)": "\n Click to expand \n",
                    "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation/ BibTex and citation info": "```\n@misc{https://doi.org/10.48550/arxiv.2201.12086,\n  doi = {10.48550/ARXIV.2201.12086},\nurl = {https://arxiv.org/abs/2201.12086},\nauthor = {Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven},\nkeywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},\ntitle = {BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},\npublisher = {arXiv},\nyear = {2022},\ncopyright = {Creative Commons Attribution 4.0 International}\n}\n```"
                }
            },
            "usage": {},
            "model_function": [
                {
                    "function_info": {
                        "return": "str",
                        "function_name": "answer_question_based_on_image",
                        "variables": [
                            {
                                "name": "image_url",
                                "type": "str",
                                "default": "./elephant.jpeg"
                            },
                            {
                                "name": "question",
                                "type": "str",
                                "default": "How many elephants are there?"
                            }
                        ]
                    }
                }
            ]
        }
    },
    "Yale-LILY/reastap-large": {
        "model_name": "reastap-large",
        "org": "Yale-LILY",
        "model_info": {
            "id": "Yale-LILY/reastap-large",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 4,
            "likes": 0,
            "library_name": "transformers",
            "tags": [
                "transformers",
                "pytorch",
                "bart",
                "text2text-generation",
                "table-question-answering",
                "table-fact-checking",
                "table-to-text",
                "en",
                "dataset:wikitablequestions",
                "dataset:wikisql",
                "dataset:tabfact",
                "dataset:logicnlg",
                "arxiv:2210.12374",
                "autotrain_compatible",
                "endpoints_compatible",
                "has_space",
                "region:us"
            ],
            "pipeline_tag": "text2text-generation",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "641376a82f36ddb7e2d54f25",
            "createdAt": "2023-03-16T20:06:00.000Z",
            "modelId": "Yale-LILY/reastap-large"
        },
        "card_to_dict": {
            "language": "en",
            "tags": [
                "table-question-answering",
                "table-fact-checking",
                "table-to-text"
            ],
            "datasets": [
                "wikitablequestions",
                "wikisql",
                "tabfact",
                "logicnlg"
            ]
        },
        "relevant_websites": [
            "https://arxiv.org/pdf/2210.12374.pdf",
            "https://github.com/Yale-LILY/ReasTAP](https://github.com/Yale-LILY/ReasTAP",
            "https://aclanthology.org/2022.emnlp-main.615",
            "https://github.com/Yale-LILY/ReasTAP"
        ],
        "text": "language: en tags: - table-question-answering - table-fact-checking - table-to-text datasets: - wikitablequestions - wikisql - tabfact - logicnlg  ReasTAP ReasTAP is a table reasoning model proposed in EMNLP 2022 paper ReasTAP: Injecting Table Reasoning Skills During Pre-training via Synthetic Reasoning Examples. The original Github repository is [). Description Yale-LILY/reastap-large (based on BART architecture) is initialized with facebook/bart-large and continuously pretrained on synthetic Table QA data to learn table structure understanding and table reasoning skills. Usage Reference bibtex @inproceedings{zhao-etal-2022-reastap,     title = \"{R}eas{TAP}: Injecting Table Reasoning Skills During Pre-training via Synthetic Reasoning Examples\",     author = \"Zhao, Yilun  and       Nan, Linyong  and       Qi, Zhenting  and       Zhang, Rui  and       Radev, Dragomir\",     booktitle = \"Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing\",     month = dec,     year = \"2022\",     address = \"Abu Dhabi, United Arab Emirates\",     publisher = \"Association for Computational Linguistics\",     url = \"\",     pages = \"9006--9018\",     abstract = \"Reasoning over tabular data requires both table structure understanding and a broad set of table reasoning skills. Current models with table-specific architectures and pre-training methods perform well on understanding table structures, but they still struggle with tasks that require various table reasoning skills. In this work, we develop ReasTAP to show that high-level table reasoning skills can be injected into models during pre-training without a complex table-specific architecture design. We define 7 table reasoning skills, such as numerical operation, temporal comparison, and conjunction. Each reasoning skill is associated with one example generator, which synthesizes questions over semi-structured tables according to the sampled templates. We model the table pre-training task as a sequence generation task and pre-train ReasTAP to generate precise answers of the synthetic examples. ReasTAP is evaluated on four benchmarks covering three downstream tasks including 1) WikiSQL-Weak and WikiTQ for Table Question Answering, 2) TabFact for Table Fact Verification, and 3) LogicNLG for Faithful Table-to-Text Generation. Experimental results demonstrate that ReasTAP achieves new state-of-the-art results on all of them and delivers a significant improvement under low-resource setting. Our code is publicly available at .\", }",
        "markdown_text": "---\nlanguage: en\ntags:\n- table-question-answering\n- table-fact-checking\n- table-to-text\ndatasets:\n- wikitablequestions\n- wikisql\n- tabfact\n- logicnlg\n---\n\n# ReasTAP\n\nReasTAP is a table reasoning model proposed in EMNLP 2022 paper [ReasTAP: Injecting Table Reasoning Skills During Pre-training via Synthetic Reasoning Examples](https://arxiv.org/pdf/2210.12374.pdf). The original Github repository is [https://github.com/Yale-LILY/ReasTAP](https://github.com/Yale-LILY/ReasTAP).\n\n## Description\n\n`Yale-LILY/reastap-large` (based on BART architecture) is initialized with `facebook/bart-large` and continuously pretrained on synthetic Table QA data to learn table structure understanding and table reasoning skills.\n\n## Usage\n\n```python\nfrom transformers import AutoTokenizer, AutoModelForSeq2SeqLM\nimport pandas as pd\n\ntokenizer = AutoTokenizer.from_pretrained(\"Yale-LILY/reastap-large\")\nmodel = AutoModelForSeq2SeqLM.from_pretrained(\"Yale-LILY/reastap-large\")\n\ndata = {\n    \"year\": [1896, 1900, 1904, 2004, 2008, 2012],\n    \"city\": [\"athens\", \"paris\", \"st. louis\", \"athens\", \"beijing\", \"london\"]\n}\ntable = pd.DataFrame.from_dict(data)\n\nquery = \"In which year did beijing host the Olympic Games?\"\nencoding = tokenizer(table=table, query=query, return_tensors=\"pt\")\n\noutputs = model.generate(**encoding)\n\nprint(tokenizer.batch_decode(outputs, skip_special_tokens=True))\n# [' 2008']\n```\n\n## Reference\n\n```bibtex\n@inproceedings{zhao-etal-2022-reastap,\n    title = \"{R}eas{TAP}: Injecting Table Reasoning Skills During Pre-training via Synthetic Reasoning Examples\",\n    author = \"Zhao, Yilun  and\n      Nan, Linyong  and\n      Qi, Zhenting  and\n      Zhang, Rui  and\n      Radev, Dragomir\",\n    booktitle = \"Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing\",\n    month = dec,\n    year = \"2022\",\n    address = \"Abu Dhabi, United Arab Emirates\",\n    publisher = \"Association for Computational Linguistics\",\n    url = \"https://aclanthology.org/2022.emnlp-main.615\",\n    pages = \"9006--9018\",\n    abstract = \"Reasoning over tabular data requires both table structure understanding and a broad set of table reasoning skills. Current models with table-specific architectures and pre-training methods perform well on understanding table structures, but they still struggle with tasks that require various table reasoning skills. In this work, we develop ReasTAP to show that high-level table reasoning skills can be injected into models during pre-training without a complex table-specific architecture design. We define 7 table reasoning skills, such as numerical operation, temporal comparison, and conjunction. Each reasoning skill is associated with one example generator, which synthesizes questions over semi-structured tables according to the sampled templates. We model the table pre-training task as a sequence generation task and pre-train ReasTAP to generate precise answers of the synthetic examples. ReasTAP is evaluated on four benchmarks covering three downstream tasks including 1) WikiSQL-Weak and WikiTQ for Table Question Answering, 2) TabFact for Table Fact Verification, and 3) LogicNLG for Faithful Table-to-Text Generation. Experimental results demonstrate that ReasTAP achieves new state-of-the-art results on all of them and delivers a significant improvement under low-resource setting. Our code is publicly available at https://github.com/Yale-LILY/ReasTAP.\",\n}\n```",
        "llm_extraction": {
            "model_name": "ReasTAP",
            "model_framework": "transformers",
            "model_architecture": "BART",
            "tasks": [
                "table-question-answering",
                "table-fact-checking",
                "table-to-text"
            ],
            "training_strategy": "synthetic reasoning examples",
            "parameters": "large",
            "vocab_size": "NONE",
            "data": "wikitablequestions, wikisql, tabfact, logicnlg",
            "authors": [
                "Zhao, Yilun",
                "Nan, Linyong",
                "Qi, Zhenting",
                "Zhang, Rui",
                "Radev, Dragomir"
            ],
            "other": [
                "EMNLP 2022 paper",
                "synthetic Table QA data",
                "table structure understanding",
                "table reasoning skills"
            ]
        },
        "truncation": 0,
        "extraction_version": "v_1",
        "papers_with_code": "successful",
        "performance": [
            {
                "id": "9446a1c0-3b92-4ed8-ac07-448d84e9695a",
                "best_rank": NaN,
                "metrics": {
                    "Test": "84.9",
                    "Val": "84.6"
                },
                "methodology": "ReasTAP-Large",
                "uses_additional_data": false,
                "paper": "reastap-injecting-table-reasoning-skills",
                "best_metric": null,
                "evaluated_on": "2022-10-22",
                "evaluation": "table-based-fact-verification-on-tabfact",
                "benchmark_details": {
                    "id": "table-based-fact-verification-on-tabfact",
                    "task": "table-based-fact-verification",
                    "dataset": "tabfact",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "f996fd00-e8e3-438f-835b-36166d9acc61",
                "best_rank": 3.0,
                "metrics": {
                    "Denotation accuracy (test)": "89.2"
                },
                "methodology": "ReasTAP-Large (weak supervision)",
                "uses_additional_data": false,
                "paper": "reastap-injecting-table-reasoning-skills",
                "best_metric": "Denotation accuracy (test)",
                "evaluated_on": "2022-10-22",
                "evaluation": "semantic-parsing-on-wikisql-1",
                "benchmark_details": {
                    "id": "semantic-parsing-on-wikisql-1",
                    "task": "semantic-parsing",
                    "dataset": "wikisql",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "2fded630-a207-40bb-b46d-06bfd1ce7602",
                "best_rank": NaN,
                "metrics": {
                    "Accuracy (Dev)": "59.7",
                    "Accuracy (Test)": "58.7"
                },
                "methodology": "ReasTAP-Large",
                "uses_additional_data": false,
                "paper": "reastap-injecting-table-reasoning-skills",
                "best_metric": null,
                "evaluated_on": "2022-10-22",
                "evaluation": "semantic-parsing-on-wikitablequestions",
                "benchmark_details": {
                    "id": "semantic-parsing-on-wikitablequestions",
                    "task": "semantic-parsing",
                    "dataset": "wikitablequestions",
                    "description": "",
                    "mirror_url": null
                }
            }
        ],
        "model_usage": {
            "llm_input": {
                "ReasTAP": {
                    "ReasTAP/ Description": "Yale-LILY/reastap-large (based on BART architecture) is initialized with facebook/bart-large and continuously pretrained on synthetic Table QA data to learn table structure understanding and table reasoning skills.",
                    "ReasTAP/ Reference": "@inproceedings{zhao-etal-2022-reastap,\n    title = \"{R}eas{TAP}: Injecting Table Reasoning Skills During Pre-training via Synthetic Reasoning Examples\",\n    author = \"Zhao, Yilun  and\n      Nan, Linyong  and\n      Qi, Zhenting  and\n      Zhang, Rui  and\n      Radev, Dragomir\",\n    booktitle = \"Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing\",\n    month = dec,\n    year = \"2022\",\n    address = \"Abu Dhabi, United Arab Emirates\",\n    publisher = \"Association for Computational Linguistics\",\n    url = \"https://aclanthology.org/2022.emnlp-main.615\",\n    pages = \"9006--9018\",\n    abstract = \"Reasoning over tabular data requires both table structure understanding and a broad set of table reasoning skills. Current models with table-specific architectures and pre-training methods perform well on understanding table structures, but they still struggle with tasks that require various table reasoning skills. In this work, we develop ReasTAP to show that high-level table reasoning skills can be injected into models during pre-training without a complex table-specific architecture design. We define 7 table reasoning skills, such as numerical operation, temporal comparison, and conjunction. Each reasoning skill is associated with one example generator, which synthesizes questions over semi-structured tables according to the sampled templates. We model the table pre-training task as a sequence generation task and pre-train ReasTAP to generate precise answers of the synthetic examples. ReasTAP is evaluated on four benchmarks covering three downstream tasks including 1) WikiSQL-Weak and WikiTQ for Table Question Answering, 2) TabFact for Table Fact Verification, and 3) LogicNLG for Faithful Table-to-Text Generation. Experimental results demonstrate that ReasTAP achieves new state-of-the-art results on all of them and delivers a significant improvement under low-resource setting. Our code is publicly available at https://github.com/Yale-LILY/ReasTAP.\",\n}"
                }
            },
            "usage": {
                "ReasTAP/ Usage": "```\nfrom transformers import AutoTokenizer, AutoModelForSeq2SeqLM\nimport pandas as pd\ntokenizer = AutoTokenizer.from_pretrained(\"Yale-LILY/reastap-large\")\nmodel = AutoModelForSeq2SeqLM.from_pretrained(\"Yale-LILY/reastap-large\")\ndata = {\n    \"year\": [1896, 1900, 1904, 2004, 2008, 2012],\n    \"city\": [\"athens\", \"paris\", \"st. louis\", \"athens\", \"beijing\", \"london\"]\n}\ntable = pd.DataFrame.from_dict(data)\nquery = \"In which year did beijing host the Olympic Games?\"\nencoding = tokenizer(table=table, query=query, return_tensors=\"pt\")\noutputs = model.generate(**encoding)\nprint(tokenizer.batch_decode(outputs, skip_special_tokens=True))\n[' 2008']\n```"
            },
            "model_function": [
                {
                    "code": "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM\nimport pandas as pd\n\ndef query_table(query, table=None, model_name=\"Yale-LILY/reastap-large\"):\n    if table is None:\n        table = {\n            \"year\": [1896, 1900, 1904, 2004, 2008, 2012],\n            \"city\": [\"athens\", \"paris\", \"st. louis\", \"athens\", \"beijing\", \"london\"]\n        }\n\n    tokenizer = AutoTokenizer.from_pretrained(model_name)\n    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)\n\n    table = pd.DataFrame.from_dict(table)\n    encoding = tokenizer(table=table, query=query, return_tensors=\"pt\")\n    outputs = model.generate(encoding)\n    return tokenizer.batch_decode(outputs, skip_special_tokens=True)\n\n# Example usage:\nquery = \"In which year did beijing host the Olympic Games?\"\nprint(query_table(query))",
                    "function_info": {
                        "return": null,
                        "function_name": "query_table"
                    }
                }
            ]
        }
    },
    "distilbert/distilbert-base-uncased-distilled-squad": {
        "model_name": "distilbert-base-uncased-distilled-squad",
        "org": "distilbert",
        "model_info": {
            "id": "distilbert/distilbert-base-uncased-distilled-squad",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 47502,
            "likes": 73,
            "library_name": "transformers",
            "tags": [
                "transformers",
                "pytorch",
                "tf",
                "tflite",
                "coreml",
                "safetensors",
                "distilbert",
                "question-answering",
                "en",
                "dataset:squad",
                "arxiv:1910.01108",
                "arxiv:1910.09700",
                "license:apache-2.0",
                "endpoints_compatible",
                "has_space",
                "region:us"
            ],
            "pipeline_tag": "question-answering",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "621ffdc036468d709f174345",
            "createdAt": "2022-03-02T23:29:04.000Z",
            "modelId": "distilbert/distilbert-base-uncased-distilled-squad"
        },
        "card_to_dict": {
            "language": "en",
            "license": "apache-2.0",
            "datasets": [
                "squad"
            ],
            "widget": [
                {
                    "text": "Which name is also used to describe the Amazon rainforest in English?",
                    "context": "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain \"Amazonas\" in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species."
                },
                {
                    "text": "How many square kilometers of rainforest is covered in the basin?",
                    "context": "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain \"Amazonas\" in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species."
                }
            ]
        },
        "relevant_websites": [
            "https://medium.com/huggingface/distilbert-8cf3380435b5",
            "https://arxiv.org/abs/1910.01108",
            "https://huggingface.co/distilbert-base-uncased",
            "https://huggingface.co/datasets/squad",
            "https://huggingface.co/distilbert-base-uncased",
            "https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation",
            "https://arxiv.org/abs/1910.01108",
            "https://aclanthology.org/2021.acl-long.330.pdf",
            "https://dl.acm.org/doi/pdf/10.1145/3442188.3445922",
            "https://huggingface.co/distilbert-base-uncased",
            "https://yknzhu.wixsite.com/mbweb",
            "https://en.wikipedia.org/wiki/English_Wikipedia",
            "https://huggingface.co/datasets/squad",
            "https://huggingface.co/distilbert-base-uncased",
            "https://huggingface.co/distilbert-base-uncased",
            "https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/README.md",
            "https://mlco2.github.io/impact#compute",
            "https://arxiv.org/abs/1910.09700",
            "https://arxiv.org/pdf/1910.01108.pdf",
            "https://arxiv.org/abs/1910.01108"
        ],
        "text": "language: en license: apache-2.0 datasets: - squad widget: - text: Which name is also used to describe the Amazon rainforest in English?   context: 'The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish:     Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch:     Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is     a moist broadleaf forest that covers most of the Amazon basin of South America.     This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which     5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This     region includes territory belonging to nine nations. The majority of the forest     is contained within Brazil, with 60% of the rainforest, followed by Peru with     13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia,     Guyana, Suriname and French Guiana. States or departments in four nations contain     \"Amazonas\" in their names. The Amazon represents over half of the planet''s remaining     rainforests, and comprises the largest and most biodiverse tract of tropical rainforest     in the world, with an estimated 390 billion individual trees divided into 16,000     species.' - text: How many square kilometers of rainforest is covered in the basin?   context: 'The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish:     Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch:     Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is     a moist broadleaf forest that covers most of the Amazon basin of South America.     This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which     5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This     region includes territory belonging to nine nations. The majority of the forest     is contained within Brazil, with 60% of the rainforest, followed by Peru with     13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia,     Guyana, Suriname and French Guiana. States or departments in four nations contain     \"Amazonas\" in their names. The Amazon represents over half of the planet''s remaining     rainforests, and comprises the largest and most biodiverse tract of tropical rainforest     in the world, with an estimated 390 billion individual trees divided into 16,000     species.'  DistilBERT base uncased distilled SQuAD Table of Contents  Model Details How To Get Started With the Model Uses Risks, Limitations and Biases Training Evaluation Environmental Impact Technical Specifications Citation Information Model Card Authors  Model Details Model Description: The DistilBERT model was proposed in the blog post Smaller, faster, cheaper, lighter: Introducing DistilBERT, adistilled version of BERT, and the paper DistilBERT, adistilled version of BERT: smaller, faster, cheaper and lighter. DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT's performances as measured on the GLUE language understanding benchmark. This model is a fine-tune checkpoint of DistilBERT-base-uncased, fine-tuned using (a second step of) knowledge distillation on SQuAD v1.1.   Developed by: Hugging Face Model Type: Transformer-based language model Language(s): English  License: Apache 2.0 Related Models: DistilBERT-base-uncased Resources for more information: See this repository for more about Distil* (a class of compressed models including this model) See Sanh et al. (2019) for more information about knowledge distillation and the training procedure  How to Get Started with the Model Use the code below to get started with the model.  Here is how to use this model in PyTorch: And in TensorFlow:  Uses This model can be used for question answering. Misuse and Out-of-scope Use The model should not be used to intentionally create hostile or alienating environments for people. In addition, the model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model. Risks, Limitations and Biases CONTENT WARNING: Readers should be aware that language generated by this model can be disturbing or offensive to some and can propagate historical and current stereotypes. Significant research has explored bias and fairness issues with language models (see, e.g., Sheng et al. (2021) and Bender et al. (2021)). Predictions generated by the model can include disturbing and harmful stereotypes across protected classes; identity characteristics; and sensitive, social, and occupational groups. For example: Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. Training Training Data The distilbert-base-uncased model model describes it's training data as:   DistilBERT pretrained on the same data as BERT, which is BookCorpus, a dataset consisting of 11,038 unpublished books and English Wikipedia (excluding lists, tables and headers).  To learn more about the SQuAD v1.1 dataset, see the SQuAD v1.1 data card. Training Procedure Preprocessing See the distilbert-base-uncased model card for further details. Pretraining See the distilbert-base-uncased model card for further details.  Evaluation As discussed in the model repository  This model reaches a F1 score of 86.9 on the [SQuAD v1.1] dev set (for comparison, Bert bert-base-uncased version reaches a F1 score of 88.5).  Environmental Impact Carbon emissions can be estimated using the Machine Learning Impact calculator presented in Lacoste et al. (2019). We present the hardware type and hours used based on the associated paper. Note that these details are just for training DistilBERT, not including the fine-tuning with SQuAD.  Hardware Type: 8 16GB V100 GPUs Hours used: 90 hours Cloud Provider: Unknown Compute Region: Unknown Carbon Emitted: Unknown  Technical Specifications See the associated paper for details on the modeling architecture, objective, compute infrastructure, and training details. Citation Information bibtex @inproceedings{sanh2019distilbert,   title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},   author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},   booktitle={NeurIPS EMC^2 Workshop},   year={2019} } APA:  - Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108. Model Card Authors This model card was written by the Hugging Face team.",
        "markdown_text": "---\nlanguage: en\nlicense: apache-2.0\ndatasets:\n- squad\nwidget:\n- text: Which name is also used to describe the Amazon rainforest in English?\n  context: 'The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish:\n    Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch:\n    Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is\n    a moist broadleaf forest that covers most of the Amazon basin of South America.\n    This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which\n    5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This\n    region includes territory belonging to nine nations. The majority of the forest\n    is contained within Brazil, with 60% of the rainforest, followed by Peru with\n    13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia,\n    Guyana, Suriname and French Guiana. States or departments in four nations contain\n    \"Amazonas\" in their names. The Amazon represents over half of the planet''s remaining\n    rainforests, and comprises the largest and most biodiverse tract of tropical rainforest\n    in the world, with an estimated 390 billion individual trees divided into 16,000\n    species.'\n- text: How many square kilometers of rainforest is covered in the basin?\n  context: 'The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish:\n    Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch:\n    Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is\n    a moist broadleaf forest that covers most of the Amazon basin of South America.\n    This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which\n    5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This\n    region includes territory belonging to nine nations. The majority of the forest\n    is contained within Brazil, with 60% of the rainforest, followed by Peru with\n    13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia,\n    Guyana, Suriname and French Guiana. States or departments in four nations contain\n    \"Amazonas\" in their names. The Amazon represents over half of the planet''s remaining\n    rainforests, and comprises the largest and most biodiverse tract of tropical rainforest\n    in the world, with an estimated 390 billion individual trees divided into 16,000\n    species.'\n---\n\n# DistilBERT base uncased distilled SQuAD\n\n## Table of Contents\n- [Model Details](#model-details)\n- [How To Get Started With the Model](#how-to-get-started-with-the-model)\n- [Uses](#uses)\n- [Risks, Limitations and Biases](#risks-limitations-and-biases)\n- [Training](#training)\n- [Evaluation](#evaluation)\n- [Environmental Impact](#environmental-impact)\n- [Technical Specifications](#technical-specifications)\n- [Citation Information](#citation-information)\n- [Model Card Authors](#model-card-authors)\n\n## Model Details\n\n**Model Description:** The DistilBERT model was proposed in the blog post [Smaller, faster, cheaper, lighter: Introducing DistilBERT, adistilled version of BERT](https://medium.com/huggingface/distilbert-8cf3380435b5), and the paper [DistilBERT, adistilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108). DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than *bert-base-uncased*, runs 60% faster while preserving over 95% of BERT's performances as measured on the GLUE language understanding benchmark.\n\nThis model is a fine-tune checkpoint of [DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased), fine-tuned using (a second step of) knowledge distillation on [SQuAD v1.1](https://huggingface.co/datasets/squad). \n\n- **Developed by:** Hugging Face\n- **Model Type:** Transformer-based language model\n- **Language(s):** English \n- **License:** Apache 2.0\n- **Related Models:** [DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased)\n- **Resources for more information:**\n  - See [this repository](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation) for more about Distil\\* (a class of compressed models including this model)\n  - See [Sanh et al. (2019)](https://arxiv.org/abs/1910.01108) for more information about knowledge distillation and the training procedure\n\n## How to Get Started with the Model \n\nUse the code below to get started with the model. \n\n```python\n>>> from transformers import pipeline\n>>> question_answerer = pipeline(\"question-answering\", model='distilbert-base-uncased-distilled-squad')\n\n>>> context = r\"\"\"\n... Extractive Question Answering is the task of extracting an answer from a text given a question. An example     of a\n... question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune\n... a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.\n... \"\"\"\n\n>>> result = question_answerer(question=\"What is a good example of a question answering dataset?\",     context=context)\n>>> print(\n... f\"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}\"\n...)\n\nAnswer: 'SQuAD dataset', score: 0.4704, start: 147, end: 160\n```\n\nHere is how to use this model in PyTorch:\n\n```python\nfrom transformers import DistilBertTokenizer, DistilBertForQuestionAnswering\nimport torch\ntokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')\nmodel = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')\n\nquestion, text = \"Who was Jim Henson?\", \"Jim Henson was a nice puppet\"\n\ninputs = tokenizer(question, text, return_tensors=\"pt\")\nwith torch.no_grad():\n    outputs = model(**inputs)\n\nanswer_start_index = torch.argmax(outputs.start_logits)\nanswer_end_index = torch.argmax(outputs.end_logits)\n\npredict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]\ntokenizer.decode(predict_answer_tokens)\n```\n\nAnd in TensorFlow: \n\n```python\nfrom transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering\nimport tensorflow as tf\n\ntokenizer = DistilBertTokenizer.from_pretrained(\"distilbert-base-uncased-distilled-squad\")\nmodel = TFDistilBertForQuestionAnswering.from_pretrained(\"distilbert-base-uncased-distilled-squad\")\n\nquestion, text = \"Who was Jim Henson?\", \"Jim Henson was a nice puppet\"\n\ninputs = tokenizer(question, text, return_tensors=\"tf\")\noutputs = model(**inputs)\n\nanswer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])\nanswer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])\n\npredict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]\ntokenizer.decode(predict_answer_tokens)\n```\n\n## Uses\n\nThis model can be used for question answering.\n\n#### Misuse and Out-of-scope Use\n\nThe model should not be used to intentionally create hostile or alienating environments for people. In addition, the model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.\n\n## Risks, Limitations and Biases\n\n**CONTENT WARNING: Readers should be aware that language generated by this model can be disturbing or offensive to some and can propagate historical and current stereotypes.**\n\nSignificant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)). Predictions generated by the model can include disturbing and harmful stereotypes across protected classes; identity characteristics; and sensitive, social, and occupational groups. For example:\n\n\n```python\n>>> from transformers import pipeline\n>>> question_answerer = pipeline(\"question-answering\", model='distilbert-base-uncased-distilled-squad')\n\n>>> context = r\"\"\"\n... Alice is sitting on the bench. Bob is sitting next to her.\n... \"\"\"\n\n>>> result = question_answerer(question=\"Who is the CEO?\", context=context)\n>>> print(\n... f\"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}\"\n...)\n\nAnswer: 'Bob', score: 0.4183, start: 32, end: 35\n```\n\nUsers (both direct and downstream) should be made aware of the risks, biases and limitations of the model.\n\n## Training\n\n#### Training Data\n\nThe [distilbert-base-uncased model](https://huggingface.co/distilbert-base-uncased) model describes it's training data as: \n\n> DistilBERT pretrained on the same data as BERT, which is [BookCorpus](https://yknzhu.wixsite.com/mbweb), a dataset consisting of 11,038 unpublished books and [English Wikipedia](https://en.wikipedia.org/wiki/English_Wikipedia) (excluding lists, tables and headers).\n\nTo learn more about the SQuAD v1.1 dataset, see the [SQuAD v1.1 data card](https://huggingface.co/datasets/squad).\n\n#### Training Procedure\n\n##### Preprocessing\n\nSee the [distilbert-base-uncased model card](https://huggingface.co/distilbert-base-uncased) for further details.\n\n##### Pretraining\n\nSee the [distilbert-base-uncased model card](https://huggingface.co/distilbert-base-uncased) for further details. \n\n## Evaluation\n\nAs discussed in the [model repository](https://github.com/huggingface/transformers/blob/main/examples/research_projects/distillation/README.md)\n\n> This model reaches a F1 score of 86.9 on the [SQuAD v1.1] dev set (for comparison, Bert bert-base-uncased version reaches a F1 score of 88.5).\n\n## Environmental Impact\n\nCarbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). We present the hardware type and hours used based on the [associated paper](https://arxiv.org/pdf/1910.01108.pdf). Note that these details are just for training DistilBERT, not including the fine-tuning with SQuAD.\n\n- **Hardware Type:** 8 16GB V100 GPUs\n- **Hours used:** 90 hours\n- **Cloud Provider:** Unknown\n- **Compute Region:** Unknown\n- **Carbon Emitted:** Unknown\n\n## Technical Specifications\n\nSee the [associated paper](https://arxiv.org/abs/1910.01108) for details on the modeling architecture, objective, compute infrastructure, and training details.\n\n## Citation Information\n\n```bibtex\n@inproceedings{sanh2019distilbert,\n  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},\n  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},\n  booktitle={NeurIPS EMC^2 Workshop},\n  year={2019}\n}\n```\n\nAPA: \n- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.\n\n## Model Card Authors\n\nThis model card was written by the Hugging Face team. \n",
        "llm_extraction": {
            "model_name": "DistilBERT",
            "model_framework": "transformers",
            "model_architecture": "distilled",
            "tasks": [
                "question-answering"
            ],
            "training_strategy": "knowledge distillation",
            "parameters": "40% less than bert-base-uncased",
            "vocab_size": "N/A",
            "training_data": "BookCorpus and English Wikipedia",
            "authors": [
                "Hugging Face"
            ],
            "other": [
                "SQuAD v1.1",
                "GLUE language understanding benchmark",
                "Apache 2.0"
            ]
        },
        "truncation": 0,
        "extraction_version": "v_1",
        "papers_with_code": "successful",
        "performance": [
            {
                "id": "cf424c6b-ef9e-4420-bd6c-37b772c84f6f",
                "best_rank": NaN,
                "metrics": {
                    "EM": "77.7"
                },
                "methodology": "DistilBERT",
                "uses_additional_data": false,
                "paper": "distilbert-a-distilled-version-of-bert",
                "best_metric": null,
                "evaluated_on": "2019-10-02",
                "evaluation": "question-answering-on-squad11-dev",
                "benchmark_details": {
                    "id": "question-answering-on-squad11-dev",
                    "task": "question-answering",
                    "dataset": "squad1-1-dev",
                    "description": "",
                    "mirror_url": null
                }
            }
        ],
        "model_usage": {
            "llm_input": {
                "DistilBERT base uncased distilled SQuAD": {
                    "DistilBERT base uncased distilled SQuAD/ Model Details": "Model Description: The DistilBERT model was proposed in the blog post Smaller, faster, cheaper, lighter: Introducing DistilBERT, adistilled version of BERT, and the paper DistilBERT, adistilled version of BERT: smaller, faster, cheaper and lighter. DistilBERT is a small, fast, cheap and light Transformer model trained by distilling BERT base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of BERT's performances as measured on the GLUE language understanding benchmark.\nThis model is a fine-tune checkpoint of DistilBERT-base-uncased, fine-tuned using (a second step of) knowledge distillation on SQuAD v1.1. \n['Developed by: Hugging Face', 'Model Type: Transformer-based language model', 'Language(s): English ', 'License: Apache 2.0', 'Related Models: DistilBERT-base-uncased', 'Resources for more information:', ['See this repository for more about Distil\\* (a class of compressed models including this model)', 'See Sanh et al. (2019) for more information about knowledge distillation and the training procedure']]",
                    "DistilBERT base uncased distilled SQuAD/ How to Get Started with the Model": "Use the code below to get started with the model. \nHere is how to use this model in PyTorch:\nAnd in TensorFlow: ",
                    "DistilBERT base uncased distilled SQuAD/ Risks, Limitations and Biases": "CONTENT WARNING: Readers should be aware that language generated by this model can be disturbing or offensive to some and can propagate historical and current stereotypes.\nSignificant research has explored bias and fairness issues with language models (see, e.g., Sheng et al. (2021) and Bender et al. (2021)). Predictions generated by the model can include disturbing and harmful stereotypes across protected classes; identity characteristics; and sensitive, social, and occupational groups. For example:\nUsers (both direct and downstream) should be made aware of the risks, biases and limitations of the model.",
                    "DistilBERT base uncased distilled SQuAD/ Training": "Training Data\nThe distilbert-base-uncased model model describes it's training data as: \n['DistilBERT pretrained on the same data as BERT, which is BookCorpus, a dataset consisting of 11,038 unpublished books and English Wikipedia (excluding lists, tables and headers).']\nTo learn more about the SQuAD v1.1 dataset, see the SQuAD v1.1 data card.\nTraining Procedure\nPreprocessing\nSee the distilbert-base-uncased model card for further details.\nPretraining\nSee the distilbert-base-uncased model card for further details. ",
                    "DistilBERT base uncased distilled SQuAD/ Environmental Impact": "Carbon emissions can be estimated using the Machine Learning Impact calculator presented in Lacoste et al. (2019). We present the hardware type and hours used based on the associated paper. Note that these details are just for training DistilBERT, not including the fine-tuning with SQuAD.\n['Hardware Type: 8 16GB V100 GPUs', 'Hours used: 90 hours', 'Cloud Provider: Unknown', 'Compute Region: Unknown', 'Carbon Emitted: Unknown']",
                    "DistilBERT base uncased distilled SQuAD/ Technical Specifications": "See the associated paper for details on the modeling architecture, objective, compute infrastructure, and training details.",
                    "DistilBERT base uncased distilled SQuAD/ Citation Information": "@inproceedings{sanh2019distilbert,\n  title={DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter},\n  author={Sanh, Victor and Debut, Lysandre and Chaumond, Julien and Wolf, Thomas},\n  booktitle={NeurIPS EMC^2 Workshop},\n  year={2019}\n}\nAPA: \n['Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.']",
                    "DistilBERT base uncased distilled SQuAD/ Model Card Authors": "This model card was written by the Hugging Face team. "
                }
            },
            "usage": {
                "DistilBERT base uncased distilled SQuAD/ Uses": "This model can be used for question answering.\nMisuse and Out-of-scope Use\nThe model should not be used to intentionally create hostile or alienating environments for people. In addition, the model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model."
            },
            "model_function": [
                {
                    "function_info": {
                        "return": "str",
                        "function_name": "question_answering",
                        "variables": [
                            {
                                "name": "question",
                                "type": "str",
                                "default": "one day I will see the world"
                            },
                            {
                                "name": "context",
                                "type": "str",
                                "default": "My name is Clara and I live in Berkeley."
                            }
                        ]
                    }
                }
            ]
        }
    },
    "dslim/bert-base-NER": {
        "model_name": "bert-base-NER",
        "org": "dslim",
        "model_info": {
            "id": "dslim/bert-base-NER",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 1193313,
            "likes": 351,
            "library_name": "transformers",
            "tags": [
                "transformers",
                "pytorch",
                "tf",
                "jax",
                "onnx",
                "safetensors",
                "bert",
                "token-classification",
                "en",
                "dataset:conll2003",
                "arxiv:1810.04805",
                "license:mit",
                "model-index",
                "autotrain_compatible",
                "endpoints_compatible",
                "has_space",
                "region:us"
            ],
            "pipeline_tag": "token-classification",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "621ffdc136468d709f17a8e5",
            "createdAt": "2022-03-02T23:29:05.000Z",
            "modelId": "dslim/bert-base-NER"
        },
        "card_to_dict": {
            "language": "en",
            "license": "mit",
            "datasets": [
                "conll2003"
            ]
        },
        "relevant_websites": [
            "https://www.aclweb.org/anthology/W03-0419.pdf",
            "https://huggingface.co/dslim/bert-large-NER",
            "https://www.aclweb.org/anthology/W03-0419.pdf",
            "https://arxiv.org/pdf/1810.04805",
            "https://github.com/google-research/bert/issues/223",
            "http://arxiv.org/abs/1810.04805",
            "https://dblp.org/rec/journals/corr/abs-1810-04805.bib",
            "https://dblp.org",
            "https://www.aclweb.org/anthology/W03-0419"
        ],
        "text": "language: en license: mit datasets: - conll2003  bert-base-NER Model description bert-base-NER is a fine-tuned BERT model that is ready to use for Named Entity Recognition and achieves state-of-the-art performance for the NER task. It has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER) and Miscellaneous (MISC).  Specifically, this model is a bert-base-cased model that was fine-tuned on the English version of the standard CoNLL-2003 Named Entity Recognition dataset.  If you'd like to use a larger BERT-large model fine-tuned on the same dataset, a bert-large-NER version is also available.  Intended uses & limitations How to use You can use this model with Transformers pipeline for NER. Limitations and bias This model is limited by its training dataset of entity-annotated news articles from a specific span of time. This may not generalize well for all use cases in different domains. Furthermore, the model occassionally tags subword tokens as entities and post-processing of results may be necessary to handle those cases.  Training data This model was fine-tuned on English version of the standard CoNLL-2003 Named Entity Recognition dataset.  The training dataset distinguishes between the beginning and continuation of an entity so that if there are back-to-back entities of the same type, the model can output where the second entity begins. As in the dataset, each token will be classified as one of the following classes: Abbreviation|Description -|- O|Outside of a named entity B-MIS |Beginning of a miscellaneous entity right after another miscellaneous entity I-MIS | Miscellaneous entity B-PER |Beginning of a person’s name right after another person’s name I-PER |Person’s name B-ORG |Beginning of an organization right after another organization I-ORG |organization B-LOC |Beginning of a location right after another location I-LOC |Location CoNLL-2003 English Dataset Statistics This dataset was derived from the Reuters corpus which consists of Reuters news stories. You can read more about how this dataset was created in the CoNLL-2003 paper.  # of training examples per entity type DatasetPER Train6600 Dev1842 Test1617 # of articles/sentences/tokens per dataset Dataset Tokens Train 203,621 Dev 51,362 Test 46,435 Training procedure This model was trained on a single NVIDIA V100 GPU with recommended hyperparameters from the original BERT paper which trained & evaluated the model on CoNLL-2003 NER task.  Eval results metrictest f1 91.3 precision 90.7 recall 91.9 The test metrics are a little lower than the official Google BERT results which encoded document context & experimented with CRF. More on replicating the original results here. BibTeX entry and citation info @article{DBLP:journals/corr/abs-1810-04805,   author    = {Jacob Devlin and                Ming{-}Wei Chang and                Kenton Lee and                Kristina Toutanova},   title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language                Understanding},   journal   = {CoRR},   volume    = {abs/1810.04805},   year      = {2018},   url       = {},   archivePrefix = {arXiv},   eprint    = {1810.04805},   timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},   biburl    = {},   bibsource = {dblp computer science bibliography, } } @inproceedings{tjong-kim-sang-de-meulder-2003-introduction,     title = \"Introduction to the {C}o{NLL}-2003 Shared Task: Language-Independent Named Entity Recognition\",     author = \"Tjong Kim Sang, Erik F.  and       De Meulder, Fien\",     booktitle = \"Proceedings of the Seventh Conference on Natural Language Learning at {HLT}-{NAACL} 2003\",     year = \"2003\",     url = \"\",     pages = \"142--147\", }",
        "markdown_text": "---\nlanguage: en\nlicense: mit\ndatasets:\n- conll2003\n---\n# bert-base-NER\n\n## Model description\n\n**bert-base-NER** is a fine-tuned BERT model that is ready to use for **Named Entity Recognition** and achieves **state-of-the-art performance** for the NER task. It has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER) and Miscellaneous (MISC). \n\nSpecifically, this model is a *bert-base-cased* model that was fine-tuned on the English version of the standard [CoNLL-2003 Named Entity Recognition](https://www.aclweb.org/anthology/W03-0419.pdf) dataset. \n\nIf you'd like to use a larger BERT-large model fine-tuned on the same dataset, a [**bert-large-NER**](https://huggingface.co/dslim/bert-large-NER/) version is also available. \n\n\n## Intended uses & limitations\n\n#### How to use\n\nYou can use this model with Transformers *pipeline* for NER.\n\n```python\nfrom transformers import AutoTokenizer, AutoModelForTokenClassification\nfrom transformers import pipeline\n\ntokenizer = AutoTokenizer.from_pretrained(\"dslim/bert-base-NER\")\nmodel = AutoModelForTokenClassification.from_pretrained(\"dslim/bert-base-NER\")\n\nnlp = pipeline(\"ner\", model=model, tokenizer=tokenizer)\nexample = \"My name is Wolfgang and I live in Berlin\"\n\nner_results = nlp(example)\nprint(ner_results)\n```\n\n#### Limitations and bias\n\nThis model is limited by its training dataset of entity-annotated news articles from a specific span of time. This may not generalize well for all use cases in different domains. Furthermore, the model occassionally tags subword tokens as entities and post-processing of results may be necessary to handle those cases. \n\n## Training data\n\nThis model was fine-tuned on English version of the standard [CoNLL-2003 Named Entity Recognition](https://www.aclweb.org/anthology/W03-0419.pdf) dataset. \n\nThe training dataset distinguishes between the beginning and continuation of an entity so that if there are back-to-back entities of the same type, the model can output where the second entity begins. As in the dataset, each token will be classified as one of the following classes:\n\nAbbreviation|Description\n-|-\nO|Outside of a named entity\nB-MIS |Beginning of a miscellaneous entity right after another miscellaneous entity\nI-MIS | Miscellaneous entity\nB-PER |Beginning of a person’s name right after another person’s name\nI-PER |Person’s name\nB-ORG |Beginning of an organization right after another organization\nI-ORG |organization\nB-LOC |Beginning of a location right after another location\nI-LOC |Location\n\n\n### CoNLL-2003 English Dataset Statistics\nThis dataset was derived from the Reuters corpus which consists of Reuters news stories. You can read more about how this dataset was created in the CoNLL-2003 paper. \n#### # of training examples per entity type\nDataset|LOC|MISC|ORG|PER\n-|-|-|-|-\nTrain|7140|3438|6321|6600\nDev|1837|922|1341|1842\nTest|1668|702|1661|1617\n#### # of articles/sentences/tokens per dataset\nDataset |Articles |Sentences |Tokens\n-|-|-|-\nTrain |946 |14,987 |203,621\nDev |216 |3,466 |51,362\nTest |231 |3,684 |46,435\n\n## Training procedure\n\nThis model was trained on a single NVIDIA V100 GPU with recommended hyperparameters from the [original BERT paper](https://arxiv.org/pdf/1810.04805) which trained & evaluated the model on CoNLL-2003 NER task. \n\n## Eval results\nmetric|dev|test\n-|-|-\nf1 |95.1 |91.3\nprecision |95.0 |90.7\nrecall |95.3 |91.9\n\nThe test metrics are a little lower than the official Google BERT results which encoded document context & experimented with CRF. More on replicating the original results [here](https://github.com/google-research/bert/issues/223).\n\n### BibTeX entry and citation info\n\n```\n@article{DBLP:journals/corr/abs-1810-04805,\n  author    = {Jacob Devlin and\n               Ming{-}Wei Chang and\n               Kenton Lee and\n               Kristina Toutanova},\n  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language\n               Understanding},\n  journal   = {CoRR},\n  volume    = {abs/1810.04805},\n  year      = {2018},\n  url       = {http://arxiv.org/abs/1810.04805},\n  archivePrefix = {arXiv},\n  eprint    = {1810.04805},\n  timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},\n  biburl    = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},\n  bibsource = {dblp computer science bibliography, https://dblp.org}\n}\n```\n```\n@inproceedings{tjong-kim-sang-de-meulder-2003-introduction,\n    title = \"Introduction to the {C}o{NLL}-2003 Shared Task: Language-Independent Named Entity Recognition\",\n    author = \"Tjong Kim Sang, Erik F.  and\n      De Meulder, Fien\",\n    booktitle = \"Proceedings of the Seventh Conference on Natural Language Learning at {HLT}-{NAACL} 2003\",\n    year = \"2003\",\n    url = \"https://www.aclweb.org/anthology/W03-0419\",\n    pages = \"142--147\",\n}\n```\n",
        "llm_extraction": {
            "model_name": "bert-base-NER",
            "model_framework": "transformers",
            "model_architecture": "BERT",
            "tasks": [
                "Named Entity Recognition"
            ],
            "training_strategy": "fine-tuning",
            "parameters": "12B",
            "vocab_size": "50,257",
            "training_data": "CoNLL-2003",
            "authors": [
                "Jacob Devlin",
                "Ming-Wei Chang",
                "Kenton Lee",
                "Kristina Toutanova"
            ],
            "other": [
                "CoNLL-2003 English Dataset",
                "BERT",
                "NER",
                "Google BERT results"
            ]
        },
        "truncation": 0,
        "extraction_version": "v_1",
        "papers_with_code": "successful",
        "performance": [
            {
                "id": "d3d7fde0-b78e-41f3-9cda-db22e357feeb",
                "best_rank": NaN,
                "metrics": {
                    "In-domain": "79.8",
                    "Out-of-domain": "74.1",
                    "Overall": "78.1"
                },
                "methodology": "BERT-base finetune (single model)",
                "uses_additional_data": false,
                "paper": "bert-pre-training-of-deep-bidirectional",
                "best_metric": null,
                "evaluated_on": "2018-10-11",
                "evaluation": "question-answering-on-coqa",
                "benchmark_details": {
                    "id": "question-answering-on-coqa",
                    "task": "question-answering",
                    "dataset": "coqa",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "4cee6d4c-fb49-4a55-ba7b-88f714a2f658",
                "best_rank": NaN,
                "metrics": {
                    "Average Accuracy": "57.52",
                    "Average Precision": "54.18",
                    "Average Recall": "54.02",
                    "Average F1": "54.10"
                },
                "methodology": "BERT",
                "uses_additional_data": false,
                "paper": "bert-pre-training-of-deep-bidirectional",
                "best_metric": null,
                "evaluated_on": "2018-10-11",
                "evaluation": "type-prediction-on-manytypes4typescript",
                "benchmark_details": {
                    "id": "type-prediction-on-manytypes4typescript",
                    "task": "type-prediction",
                    "dataset": "manytypes4typescript",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "8ab42d26-ddb8-4209-95e3-056b933e8fc7",
                "best_rank": NaN,
                "metrics": {
                    "F1": "86.37"
                },
                "methodology": "BERT Base",
                "uses_additional_data": false,
                "paper": "bert-pre-training-of-deep-bidirectional",
                "best_metric": null,
                "evaluated_on": "2018-10-11",
                "evaluation": "named-entity-recognition-on-ncbi-disease-1",
                "benchmark_details": {
                    "id": "named-entity-recognition-on-ncbi-disease-1",
                    "task": "named-entity-recognition-1",
                    "dataset": "ncbi-disease",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "e2d4b079-feca-4e80-8d94-200ddf9ff46c",
                "best_rank": NaN,
                "metrics": {
                    "F1": "53.2",
                    "Precision": "56.1",
                    "Recall": "50.6"
                },
                "methodology": "BERT",
                "uses_additional_data": false,
                "paper": "bert-pre-training-of-deep-bidirectional",
                "best_metric": null,
                "evaluated_on": "2018-10-11",
                "evaluation": "multimodal-intent-recognition-on-photochat",
                "benchmark_details": {
                    "id": "multimodal-intent-recognition-on-photochat",
                    "task": "multimodal-intent-recognition",
                    "dataset": "photochat",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "f4d9926c-114f-4b0f-acf3-3a437f4be501",
                "best_rank": NaN,
                "metrics": {
                    "F1": "56.065",
                    "EM": "54.040"
                },
                "methodology": "BERT-Base (single model)",
                "uses_additional_data": false,
                "paper": "bert-pre-training-of-deep-bidirectional",
                "best_metric": null,
                "evaluated_on": "2018-10-11",
                "evaluation": "common-sense-reasoning-on-record",
                "benchmark_details": {
                    "id": "common-sense-reasoning-on-record",
                    "task": "common-sense-reasoning",
                    "dataset": "record",
                    "description": "This page is mirroring [ReCoRD Leaderboard](https://sheng-z.github.io/ReCoRD-explorer/).",
                    "mirror_url": null
                }
            },
            {
                "id": "cd9d30e5-ee2b-416d-a90c-7e11c8258c4c",
                "best_rank": NaN,
                "metrics": {
                    "F1": "65.24"
                },
                "methodology": "BERT Base",
                "uses_additional_data": false,
                "paper": "bert-pre-training-of-deep-bidirectional",
                "best_metric": null,
                "evaluated_on": "2018-10-11",
                "evaluation": "named-entity-recognition-on-scierc",
                "benchmark_details": {
                    "id": "named-entity-recognition-on-scierc",
                    "task": "named-entity-recognition-1",
                    "dataset": "scierc",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "279670d3-a036-407f-9d9b-c3646743e00e",
                "best_rank": NaN,
                "metrics": {
                    "Accuracy": "84.9"
                },
                "methodology": "BERT",
                "uses_additional_data": false,
                "paper": "bert-pre-training-of-deep-bidirectional",
                "best_metric": null,
                "evaluated_on": "2018-10-11",
                "evaluation": "linear-probe-classification-on-senteval",
                "benchmark_details": {
                    "id": "linear-probe-classification-on-senteval",
                    "task": "linear-probe-classification",
                    "dataset": "senteval",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "8c807034-a718-439b-8b35-60d50e77ceb0",
                "best_rank": 1.0,
                "metrics": {
                    "Accuracy": "70.5%"
                },
                "methodology": "BERT",
                "uses_additional_data": false,
                "paper": "bert-pre-training-of-deep-bidirectional",
                "best_metric": "Accuracy",
                "evaluated_on": "2018-10-11",
                "evaluation": "cross-lingual-natural-language-inference-on-3",
                "benchmark_details": {
                    "id": "cross-lingual-natural-language-inference-on-3",
                    "task": "cross-lingual-natural-language-inference",
                    "dataset": "xnli-zero-shot-english-to-german",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "725a1927-67c1-402a-b56c-37e003cc75c4",
                "best_rank": NaN,
                "metrics": {
                    "Accuracy": "74.3%"
                },
                "methodology": "BERT",
                "uses_additional_data": false,
                "paper": "bert-pre-training-of-deep-bidirectional",
                "best_metric": null,
                "evaluated_on": "2018-10-11",
                "evaluation": "cross-lingual-natural-language-inference-on-1",
                "benchmark_details": {
                    "id": "cross-lingual-natural-language-inference-on-1",
                    "task": "cross-lingual-natural-language-inference",
                    "dataset": "xnli-zero-shot-english-to-spanish",
                    "description": "",
                    "mirror_url": null
                }
            }
        ],
        "model_usage": {
            "llm_input": {
                "bert-base-NER": {
                    "bert-base-NER/ Model description": "bert-base-NER is a fine-tuned BERT model that is ready to use for Named Entity Recognition and achieves state-of-the-art performance for the NER task. It has been trained to recognize four types of entities: location (LOC), organizations (ORG), person (PER) and Miscellaneous (MISC). \nSpecifically, this model is a bert-base-cased model that was fine-tuned on the English version of the standard CoNLL-2003 Named Entity Recognition dataset. \nIf you'd like to use a larger BERT-large model fine-tuned on the same dataset, a bert-large-NER version is also available. ",
                    "bert-base-NER/ Intended uses & limitations": "How to use\nYou can use this model with Transformers pipeline for NER.\nLimitations and bias\nThis model is limited by its training dataset of entity-annotated news articles from a specific span of time. This may not generalize well for all use cases in different domains. Furthermore, the model occassionally tags subword tokens as entities and post-processing of results may be necessary to handle those cases. ",
                    "CoNLL-2003 English Dataset Statistics/ # of training examples per entity type": "Dataset|LOC|MISC|ORG|PER\n-|-|-|-|-\nTrain|7140|3438|6321|6600\nDev|1837|922|1341|1842\nTest|1668|702|1661|1617",
                    "CoNLL-2003 English Dataset Statistics/ # of articles/sentences/tokens per dataset": "Dataset |Articles |Sentences |Tokens\n-|-|-|-\nTrain |946 |14,987 |203,621\nDev |216 |3,466 |51,362\nTest |231 |3,684 |46,435",
                    "bert-base-NER/ Training procedure": "This model was trained on a single NVIDIA V100 GPU with recommended hyperparameters from the original BERT paper which trained & evaluated the model on CoNLL-2003 NER task. ",
                    "Eval results/ BibTeX entry and citation info": "@article{DBLP:journals/corr/abs-1810-04805,\n  author    = {Jacob Devlin and\n               Ming{-}Wei Chang and\n               Kenton Lee and\n               Kristina Toutanova},\n  title     = {{BERT:} Pre-training of Deep Bidirectional Transformers for Language\n               Understanding},\n  journal   = {CoRR},\n  volume    = {abs/1810.04805},\n  year      = {2018},\n  url       = {http://arxiv.org/abs/1810.04805},\n  archivePrefix = {arXiv},\n  eprint    = {1810.04805},\n  timestamp = {Tue, 30 Oct 2018 20:39:56 +0100},\n  biburl    = {https://dblp.org/rec/journals/corr/abs-1810-04805.bib},\n  bibsource = {dblp computer science bibliography, https://dblp.org}\n}\n@inproceedings{tjong-kim-sang-de-meulder-2003-introduction,\n    title = \"Introduction to the {C}o{NLL}-2003 Shared Task: Language-Independent Named Entity Recognition\",\n    author = \"Tjong Kim Sang, Erik F.  and\n      De Meulder, Fien\",\n    booktitle = \"Proceedings of the Seventh Conference on Natural Language Learning at {HLT}-{NAACL} 2003\",\n    year = \"2003\",\n    url = \"https://www.aclweb.org/anthology/W03-0419\",\n    pages = \"142--147\",\n}"
                }
            },
            "usage": {
                "bert-base-NER/ Intended uses & limitations": "How to use\nYou can use this model with Transformers pipeline for NER.\n```\nfrom transformers import AutoTokenizer, AutoModelForTokenClassification\nfrom transformers import pipeline\ntokenizer = AutoTokenizer.from_pretrained(\"dslim/bert-base-NER\")\nmodel = AutoModelForTokenClassification.from_pretrained(\"dslim/bert-base-NER\")\nnlp = pipeline(\"ner\", model=model, tokenizer=tokenizer)\nexample = \"My name is Wolfgang and I live in Berlin\"\nner_results = nlp(example)\nprint(ner_results)\n```\nLimitations and bias\nThis model is limited by its training dataset of entity-annotated news articles from a specific span of time. This may not generalize well for all use cases in different domains. Furthermore, the model occassionally tags subword tokens as entities and post-processing of results may be necessary to handle those cases. "
            },
            "model_function": [
                {
                    "code": "from transformers import AutoTokenizer, AutoModelForTokenClassification\nfrom transformers import pipeline\n\ndef named_entity_recognition(example, model_name=\"dslim/bert-base-NER\"):\n    tokenizer = AutoTokenizer.from_pretrained(model_name)\n    model = AutoModelForTokenClassification.from_pretrained(model_name)\n    nlp = pipeline(\"ner\", model=model, tokenizer=tokenizer)\n    ner_results = nlp(example)\n    return ner_results\n\n# Example usage\nexample = \"My name is Wolfgang and I live in Berlin\"\nner_results = named_entity_recognition(example)\nprint(ner_results)",
                    "function_info": {
                        "return": null,
                        "function_name": "named_entity_recognition",
                        "variables": [
                            {
                                "name": "text",
                                "type": "str",
                                "default": "John Cena"
                            }
                        ]
                    }
                }
            ]
        }
    },
    "facebook/bart-large-cnn": {
        "model_name": "bart-large-cnn",
        "org": "facebook",
        "model_info": {
            "id": "facebook/bart-large-cnn",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 2385096,
            "likes": 801,
            "library_name": "transformers",
            "tags": [
                "transformers",
                "pytorch",
                "tf",
                "jax",
                "rust",
                "safetensors",
                "bart",
                "text2text-generation",
                "summarization",
                "en",
                "dataset:cnn_dailymail",
                "arxiv:1910.13461",
                "license:mit",
                "model-index",
                "autotrain_compatible",
                "endpoints_compatible",
                "has_space",
                "region:us"
            ],
            "pipeline_tag": "summarization",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "621ffdc136468d709f17adb6",
            "createdAt": "2022-03-02T23:29:05.000Z",
            "modelId": "facebook/bart-large-cnn"
        },
        "card_to_dict": {
            "language": [
                "en"
            ],
            "license": "mit",
            "tags": [
                "summarization"
            ],
            "datasets": [
                "cnn_dailymail"
            ],
            "thumbnail": "https://huggingface.co/front/thumbnails/facebook.png",
            "model-index": [
                {
                    "name": "facebook/bart-large-cnn",
                    "results": [
                        {
                            "task": {
                                "type": "summarization",
                                "name": "Summarization"
                            },
                            "dataset": {
                                "name": "cnn_dailymail",
                                "type": "cnn_dailymail",
                                "config": "3.0.0",
                                "split": "train"
                            },
                            "metrics": [
                                {
                                    "type": "rouge",
                                    "value": 42.9486,
                                    "name": "ROUGE-1",
                                    "verified": true
                                },
                                {
                                    "type": "rouge",
                                    "value": 20.8149,
                                    "name": "ROUGE-2",
                                    "verified": true
                                },
                                {
                                    "type": "rouge",
                                    "value": 30.6186,
                                    "name": "ROUGE-L",
                                    "verified": true
                                },
                                {
                                    "type": "rouge",
                                    "value": 40.0376,
                                    "name": "ROUGE-LSUM",
                                    "verified": true
                                },
                                {
                                    "type": "loss",
                                    "value": 2.529000997543335,
                                    "name": "loss",
                                    "verified": true
                                },
                                {
                                    "type": "gen_len",
                                    "value": 78.5866,
                                    "name": "gen_len",
                                    "verified": true
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        "relevant_websites": [
            "https://huggingface.co/front/thumbnails/facebook.png",
            "https://huggingface.co/datasets/cnn_dailymail",
            "https://arxiv.org/abs/1910.13461",
            "https://github.com/pytorch/fairseq/tree/master/examples/bart",
            "https://huggingface.co/transformers/main_classes/pipelines.html",
            "http://arxiv.org/abs/1910.13461",
            "https://dblp.org/rec/journals/corr/abs-1910-13461.bib",
            "https://dblp.org"
        ],
        "text": "language: - en license: mit tags: - summarization datasets: - cnn_dailymail thumbnail:  model-index: - name: facebook/bart-large-cnn   results:   - task:       type: summarization       name: Summarization     dataset:       name: cnn_dailymail       type: cnn_dailymail       config: 3.0.0       split: train     metrics:     - type: rouge       value: 42.9486       name: ROUGE-1       verified: true     - type: rouge       value: 20.8149       name: ROUGE-2       verified: true     - type: rouge       value: 30.6186       name: ROUGE-L       verified: true     - type: rouge       value: 40.0376       name: ROUGE-LSUM       verified: true     - type: loss       value: 2.529000997543335       name: loss       verified: true     - type: gen_len       value: 78.5866       name: gen_len       verified: true  BART (large-sized model), fine-tuned on CNN Daily Mail BART model pre-trained on English language, and fine-tuned on CNN Daily Mail. It was introduced in the paper BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension by Lewis et al. and first released in [this repository ().  Disclaimer: The team releasing BART did not write a model card for this model so this model card has been written by the Hugging Face team. Model description BART is a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder. BART is pre-trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text. BART is particularly effective when fine-tuned for text generation (e.g. summarization, translation) but also works well for comprehension tasks (e.g. text classification, question answering). This particular checkpoint has been fine-tuned on CNN Daily Mail, a large collection of text-summary pairs. Intended uses & limitations You can use this model for text summarization.  How to use Here is how to use this model with the pipeline API: BibTeX entry and citation info bibtex @article{DBLP:journals/corr/abs-1910-13461,   author    = {Mike Lewis and                Yinhan Liu and                Naman Goyal and                Marjan Ghazvininejad and                Abdelrahman Mohamed and                Omer Levy and                Veselin Stoyanov and                Luke Zettlemoyer},   title     = {{BART:} Denoising Sequence-to-Sequence Pre-training for Natural Language                Generation, Translation, and Comprehension},   journal   = {CoRR},   volume    = {abs/1910.13461},   year      = {2019},   url       = {},   eprinttype = {arXiv},   eprint    = {1910.13461},   timestamp = {Thu, 31 Oct 2019 14:02:26 +0100},   biburl    = {},   bibsource = {dblp computer science bibliography, } }",
        "markdown_text": "---\nlanguage:\n- en\nlicense: mit\ntags:\n- summarization\ndatasets:\n- cnn_dailymail\nthumbnail: https://huggingface.co/front/thumbnails/facebook.png\nmodel-index:\n- name: facebook/bart-large-cnn\n  results:\n  - task:\n      type: summarization\n      name: Summarization\n    dataset:\n      name: cnn_dailymail\n      type: cnn_dailymail\n      config: 3.0.0\n      split: train\n    metrics:\n    - type: rouge\n      value: 42.9486\n      name: ROUGE-1\n      verified: true\n    - type: rouge\n      value: 20.8149\n      name: ROUGE-2\n      verified: true\n    - type: rouge\n      value: 30.6186\n      name: ROUGE-L\n      verified: true\n    - type: rouge\n      value: 40.0376\n      name: ROUGE-LSUM\n      verified: true\n    - type: loss\n      value: 2.529000997543335\n      name: loss\n      verified: true\n    - type: gen_len\n      value: 78.5866\n      name: gen_len\n      verified: true\n---\n# BART (large-sized model), fine-tuned on CNN Daily Mail \n\nBART model pre-trained on English language, and fine-tuned on [CNN Daily Mail](https://huggingface.co/datasets/cnn_dailymail). It was introduced in the paper [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) by Lewis et al. and first released in [this repository (https://github.com/pytorch/fairseq/tree/master/examples/bart). \n\nDisclaimer: The team releasing BART did not write a model card for this model so this model card has been written by the Hugging Face team.\n\n## Model description\n\nBART is a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder. BART is pre-trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text.\n\nBART is particularly effective when fine-tuned for text generation (e.g. summarization, translation) but also works well for comprehension tasks (e.g. text classification, question answering). This particular checkpoint has been fine-tuned on CNN Daily Mail, a large collection of text-summary pairs.\n\n## Intended uses & limitations\n\nYou can use this model for text summarization. \n\n### How to use\n\nHere is how to use this model with the [pipeline API](https://huggingface.co/transformers/main_classes/pipelines.html):\n\n```python\nfrom transformers import pipeline\n\nsummarizer = pipeline(\"summarization\", model=\"facebook/bart-large-cnn\")\n\nARTICLE = \"\"\" New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.\nA year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.\nOnly 18 days after that marriage, she got hitched yet again. Then, Barrientos declared \"I do\" five more times, sometimes only within two weeks of each other.\nIn 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her \"first and only\" marriage.\nBarrientos, now 39, is facing two criminal counts of \"offering a false instrument for filing in the first degree,\" referring to her false statements on the\n2010 marriage license application, according to court documents.\nProsecutors said the marriages were part of an immigration scam.\nOn Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.\nAfter leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective\nAnnette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.\nAll occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.\nProsecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.\nAny divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.\nThe case was referred to the Bronx District Attorney\\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\\'s\nInvestigation Division. Seven of the men are from so-called \"red-flagged\" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.\nHer eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.\nIf convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.\n\"\"\"\nprint(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))\n>>> [{'summary_text': 'Liana Barrientos, 39, is charged with two counts of \"offering a false instrument for filing in the first degree\" In total, she has been married 10 times, with nine of her marriages occurring between 1999 and 2002. She is believed to still be married to four men.'}]\n```\n\n### BibTeX entry and citation info\n\n```bibtex\n@article{DBLP:journals/corr/abs-1910-13461,\n  author    = {Mike Lewis and\n               Yinhan Liu and\n               Naman Goyal and\n               Marjan Ghazvininejad and\n               Abdelrahman Mohamed and\n               Omer Levy and\n               Veselin Stoyanov and\n               Luke Zettlemoyer},\n  title     = {{BART:} Denoising Sequence-to-Sequence Pre-training for Natural Language\n               Generation, Translation, and Comprehension},\n  journal   = {CoRR},\n  volume    = {abs/1910.13461},\n  year      = {2019},\n  url       = {http://arxiv.org/abs/1910.13461},\n  eprinttype = {arXiv},\n  eprint    = {1910.13461},\n  timestamp = {Thu, 31 Oct 2019 14:02:26 +0100},\n  biburl    = {https://dblp.org/rec/journals/corr/abs-1910-13461.bib},\n  bibsource = {dblp computer science bibliography, https://dblp.org}\n}\n```",
        "llm_extraction": {
            "model_name": "facebook/bart-large-cnn",
            "model_framework": "transformers",
            "model_architecture": "encoder-decoder",
            "tasks": [
                "summarization"
            ],
            "training_strategy": "denoising sequence-to-sequence pre-training",
            "parameters": "large-sized",
            "vocab_size": "NONE",
            "training_data": "CNN Daily Mail",
            "authors": [
                "Mike Lewis",
                "Yinhan Liu",
                "Naman Goyal",
                "Marjan Ghazvininejad",
                "Abdelrahman Mohamed",
                "Omer Levy",
                "Veselin Stoyanov",
                "Luke Zettlemoyer"
            ],
            "other": [
                "BERT-like",
                "GPT-like",
                "seq2seq",
                "corrupting text",
                "text generation",
                "comprehension tasks",
                "text classification",
                "question answering"
            ]
        },
        "truncation": 0,
        "extraction_version": "v_1",
        "papers_with_code": "successful",
        "performance": [
            {
                "id": "709faa47-7f86-4d76-a33c-0bee0a4da542",
                "best_rank": NaN,
                "metrics": {
                    "ROUGE-1": "44.16",
                    "ROUGE-2": "21.28",
                    "ROUGE-L": "40.90"
                },
                "methodology": "BART",
                "uses_additional_data": false,
                "paper": "bart-denoising-sequence-to-sequence-pre",
                "best_metric": null,
                "evaluated_on": "2019-10-29",
                "evaluation": "abstractive-text-summarization-on-cnn-daily",
                "benchmark_details": {
                    "id": "abstractive-text-summarization-on-cnn-daily",
                    "task": "abstractive-text-summarization",
                    "dataset": "cnn-daily-mail",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "d346a6c7-6cfc-4357-aa6a-e8657d6f03f9",
                "best_rank": 3.0,
                "metrics": {
                    "Rouge-L": "24.3",
                    "Rouge-1": "30.6",
                    "Rouge-2": "6.2"
                },
                "methodology": "BART",
                "uses_additional_data": false,
                "paper": "bart-denoising-sequence-to-sequence-pre",
                "best_metric": "Rouge-L",
                "evaluated_on": "2019-10-29",
                "evaluation": "open-domain-question-answering-on-eli5",
                "benchmark_details": {
                    "id": "open-domain-question-answering-on-eli5",
                    "task": "open-domain-question-answering",
                    "dataset": "eli5",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "131de994-12ce-46e5-8183-2c0a55cea548",
                "best_rank": 7.0,
                "metrics": {
                    "ROUGE-1": "45.14",
                    "ROUGE-2": "22.27",
                    "ROUGE-3": "37.25"
                },
                "methodology": "BART",
                "uses_additional_data": false,
                "paper": "bart-denoising-sequence-to-sequence-pre",
                "best_metric": "ROUGE-1",
                "evaluated_on": "2019-10-29",
                "evaluation": "text-summarization-on-x-sum",
                "benchmark_details": {
                    "id": "text-summarization-on-x-sum",
                    "task": "text-summarization",
                    "dataset": "x-sum",
                    "description": "",
                    "mirror_url": null
                }
            }
        ],
        "model_usage": {
            "llm_input": {
                "BART (large-sized model), fine-tuned on CNN Daily Mail ": {
                    "BART (large-sized model), fine-tuned on CNN Daily Mail / Model description": "BART is a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder. BART is pre-trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text.\nBART is particularly effective when fine-tuned for text generation (e.g. summarization, translation) but also works well for comprehension tasks (e.g. text classification, question answering). This particular checkpoint has been fine-tuned on CNN Daily Mail, a large collection of text-summary pairs.",
                    "Intended uses & limitations/ BibTeX entry and citation info": "@article{DBLP:journals/corr/abs-1910-13461,\n  author    = {Mike Lewis and\n               Yinhan Liu and\n               Naman Goyal and\n               Marjan Ghazvininejad and\n               Abdelrahman Mohamed and\n               Omer Levy and\n               Veselin Stoyanov and\n               Luke Zettlemoyer},\n  title     = {{BART:} Denoising Sequence-to-Sequence Pre-training for Natural Language\n               Generation, Translation, and Comprehension},\n  journal   = {CoRR},\n  volume    = {abs/1910.13461},\n  year      = {2019},\n  url       = {http://arxiv.org/abs/1910.13461},\n  eprinttype = {arXiv},\n  eprint    = {1910.13461},\n  timestamp = {Thu, 31 Oct 2019 14:02:26 +0100},\n  biburl    = {https://dblp.org/rec/journals/corr/abs-1910-13461.bib},\n  bibsource = {dblp computer science bibliography, https://dblp.org}\n}"
                }
            },
            "usage": {
                "Intended uses & limitations/ How to use": "Here is how to use this model with the pipeline API:\n```\nfrom transformers import pipeline\nsummarizer = pipeline(\"summarization\", model=\"facebook/bart-large-cnn\")\nARTICLE = \"\"\" New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.\nA year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.\nOnly 18 days after that marriage, she got hitched yet again. Then, Barrientos declared \"I do\" five more times, sometimes only within two weeks of each other.\nIn 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her \"first and only\" marriage.\nBarrientos, now 39, is facing two criminal counts of \"offering a false instrument for filing in the first degree,\" referring to her false statements on the\n2010 marriage license application, according to court documents.\nProsecutors said the marriages were part of an immigration scam.\nOn Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.\nAfter leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective\nAnnette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.\nAll occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.\nProsecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.\nAny divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.\nThe case was referred to the Bronx District Attorney\\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\\'s\nInvestigation Division. Seven of the men are from so-called \"red-flagged\" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.\nHer eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.\nIf convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.\n\"\"\"\nprint(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))\n\n\n\n[{'summary_text': 'Liana Barrientos, 39, is charged with two counts of \"offering a false instrument for filing in the first degree\" In total, she has been married 10 times, with nine of her marriages occurring between 1999 and 2002. She is believed to still be married to four men.'}]\n```\n\n\n"
            },
            "model_function": [
                {
                    "code": "from transformers import pipeline\n\ndef summarize_text(text, model=\"facebook/bart-large-cnn\", max_length=130, min_length=30, do_sample=False):\n    summarizer = pipeline(\"summarization\", model=model)\n    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)\n\nARTICLE = \"\"\"\nNew York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.\nA year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.\nOnly 18 days after that marriage, she got hitched yet again. Then, Barrientos declared \\\"I do\\\" five more times, sometimes only within two weeks of each other.\nIn 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her \\\"first and only\\\" marriage.\nBarrientos, now 39, is facing two criminal counts of \\\"offering a false instrument for filing in the first degree,\\\" referring to her 2010 marriage license application, according to court documents.\nProsecutors said the marriages were part of an immigration scam.\nOn Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.\nAfter leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective\nAnnette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.\nAll occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.\nProsecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.\nAny divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.\nThe case was referred to the Bronx District Attorney's Office by Immigration and Customs Enforcement and the Department of Homeland Security's\nInvestigation Division. Seven of the men are from so-called \\\"red-flagged\\\" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.\nHer eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.\nIf convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.\n\"\"\"\n\nprint(summarize_text(ARTICLE))",
                    "function_info": {
                        "return": null,
                        "function_name": "summarize_text"
                    }
                }
            ]
        }
    },
    "facebook/bart-large-mnli": {
        "model_name": "bart-large-mnli",
        "org": "facebook",
        "model_info": {
            "id": "facebook/bart-large-mnli",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 4708360,
            "likes": 841,
            "library_name": "transformers",
            "tags": [
                "transformers",
                "pytorch",
                "jax",
                "rust",
                "safetensors",
                "bart",
                "text-classification",
                "zero-shot-classification",
                "dataset:multi_nli",
                "arxiv:1910.13461",
                "arxiv:1909.00161",
                "license:mit",
                "autotrain_compatible",
                "endpoints_compatible",
                "has_space",
                "region:us"
            ],
            "pipeline_tag": "zero-shot-classification",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "621ffdc136468d709f17adb7",
            "createdAt": "2022-03-02T23:29:05.000Z",
            "modelId": "facebook/bart-large-mnli"
        },
        "card_to_dict": {
            "license": "mit",
            "datasets": [
                "multi_nli"
            ],
            "thumbnail": "https://huggingface.co/front/thumbnails/facebook.png",
            "pipeline_tag": "zero-shot-classification"
        },
        "relevant_websites": [
            "https://huggingface.co/front/thumbnails/facebook.png",
            "https://huggingface.co/facebook/bart-large",
            "https://huggingface.co/datasets/multi_nli",
            "https://huggingface.co/facebook/bart-large",
            "https://arxiv.org/abs/1910.13461",
            "https://github.com/pytorch/fairseq/tree/master/fairseq/models/bart",
            "https://arxiv.org/abs/1909.00161",
            "https://joeddav.github.io/blog/2020/05/29/ZSL.html"
        ],
        "text": "license: mit datasets: - multi_nli thumbnail:  pipeline_tag: zero-shot-classification  bart-large-mnli This is the checkpoint for bart-large after being trained on the MultiNLI (MNLI) dataset. Additional information about this model: - The bart-large model page - BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension  - BART fairseq implementation NLI-based Zero Shot Text Classification Yin et al. proposed a method for using pre-trained NLI models as a ready-made zero-shot sequence classifiers. The method works by posing the sequence to be classified as the NLI premise and to construct a hypothesis from each candidate label. For example, if we want to evaluate whether a sequence belongs to the class \"politics\", we could construct a hypothesis of This text is about politics.. The probabilities for entailment and contradiction are then converted to label probabilities. This method is surprisingly effective in many cases, particularly when used with larger pre-trained models like BART and Roberta. See this blog post for a more expansive introduction to this and other zero shot methods, and see the code snippets below for examples of using this model for zero-shot classification both with Hugging Face's built-in pipeline and with native Transformers/PyTorch code. With the zero-shot classification pipeline The model can be loaded with the zero-shot-classification pipeline like so: You can then use this pipeline to classify sequences into any of the class names you specify. If more than one candidate label can be correct, pass multi_label=True to calculate each class independently: With manual PyTorch",
        "markdown_text": "---\nlicense: mit\ndatasets:\n- multi_nli\nthumbnail: https://huggingface.co/front/thumbnails/facebook.png\npipeline_tag: zero-shot-classification\n---\n\n# bart-large-mnli\n\nThis is the checkpoint for [bart-large](https://huggingface.co/facebook/bart-large) after being trained on the [MultiNLI (MNLI)](https://huggingface.co/datasets/multi_nli) dataset.\n\nAdditional information about this model:\n- The [bart-large](https://huggingface.co/facebook/bart-large) model page\n- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension\n](https://arxiv.org/abs/1910.13461)\n- [BART fairseq implementation](https://github.com/pytorch/fairseq/tree/master/fairseq/models/bart)\n\n## NLI-based Zero Shot Text Classification\n\n[Yin et al.](https://arxiv.org/abs/1909.00161) proposed a method for using pre-trained NLI models as a ready-made zero-shot sequence classifiers. The method works by posing the sequence to be classified as the NLI premise and to construct a hypothesis from each candidate label. For example, if we want to evaluate whether a sequence belongs to the class \"politics\", we could construct a hypothesis of `This text is about politics.`. The probabilities for entailment and contradiction are then converted to label probabilities.\n\nThis method is surprisingly effective in many cases, particularly when used with larger pre-trained models like BART and Roberta. See [this blog post](https://joeddav.github.io/blog/2020/05/29/ZSL.html) for a more expansive introduction to this and other zero shot methods, and see the code snippets below for examples of using this model for zero-shot classification both with Hugging Face's built-in pipeline and with native Transformers/PyTorch code.\n\n#### With the zero-shot classification pipeline\n\nThe model can be loaded with the `zero-shot-classification` pipeline like so:\n\n```python\nfrom transformers import pipeline\nclassifier = pipeline(\"zero-shot-classification\",\n                      model=\"facebook/bart-large-mnli\")\n```\n\nYou can then use this pipeline to classify sequences into any of the class names you specify.\n\n```python\nsequence_to_classify = \"one day I will see the world\"\ncandidate_labels = ['travel', 'cooking', 'dancing']\nclassifier(sequence_to_classify, candidate_labels)\n#{'labels': ['travel', 'dancing', 'cooking'],\n# 'scores': [0.9938651323318481, 0.0032737774308770895, 0.002861034357920289],\n# 'sequence': 'one day I will see the world'}\n```\n\nIf more than one candidate label can be correct, pass `multi_label=True` to calculate each class independently:\n\n```python\ncandidate_labels = ['travel', 'cooking', 'dancing', 'exploration']\nclassifier(sequence_to_classify, candidate_labels, multi_label=True)\n#{'labels': ['travel', 'exploration', 'dancing', 'cooking'],\n# 'scores': [0.9945111274719238,\n#  0.9383890628814697,\n#  0.0057061901316046715,\n#  0.0018193122232332826],\n# 'sequence': 'one day I will see the world'}\n```\n\n\n#### With manual PyTorch\n\n```python\n# pose sequence as a NLI premise and label as a hypothesis\nfrom transformers import AutoModelForSequenceClassification, AutoTokenizer\nnli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')\ntokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')\n\npremise = sequence\nhypothesis = f'This example is {label}.'\n\n# run through model pre-trained on MNLI\nx = tokenizer.encode(premise, hypothesis, return_tensors='pt',\n                     truncation_strategy='only_first')\nlogits = nli_model(x.to(device))[0]\n\n# we throw away \"neutral\" (dim 1) and take the probability of\n# \"entailment\" (2) as the probability of the label being true \nentail_contradiction_logits = logits[:,[0,2]]\nprobs = entail_contradiction_logits.softmax(dim=1)\nprob_label_is_true = probs[:,1]\n```\n",
        "llm_extraction": {
            "model_name": "bart-large-mnli",
            "model_framework": "transformers",
            "model_architecture": "denoising sequence-to-sequence pre-training",
            "tasks": [
                "zero-shot-classification"
            ],
            "training_strategy": "NLI-based Zero Shot Text Classification",
            "parameters": "large",
            "vocab_size": "NONE",
            "training_data": "MultiNLI (MNLI) dataset",
            "authors": [
                "Yin et al."
            ],
            "other": [
                "BART fairseq implementation",
                "blog post",
                "code snippets"
            ]
        },
        "truncation": 0,
        "extraction_version": "v_1",
        "papers_with_code": "successful",
        "performance": [
            {
                "id": "709faa47-7f86-4d76-a33c-0bee0a4da542",
                "best_rank": NaN,
                "metrics": {
                    "ROUGE-1": "44.16",
                    "ROUGE-2": "21.28",
                    "ROUGE-L": "40.90"
                },
                "methodology": "BART",
                "uses_additional_data": false,
                "paper": "bart-denoising-sequence-to-sequence-pre",
                "best_metric": null,
                "evaluated_on": "2019-10-29",
                "evaluation": "abstractive-text-summarization-on-cnn-daily",
                "benchmark_details": {
                    "id": "abstractive-text-summarization-on-cnn-daily",
                    "task": "abstractive-text-summarization",
                    "dataset": "cnn-daily-mail",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "d346a6c7-6cfc-4357-aa6a-e8657d6f03f9",
                "best_rank": 3.0,
                "metrics": {
                    "Rouge-L": "24.3",
                    "Rouge-1": "30.6",
                    "Rouge-2": "6.2"
                },
                "methodology": "BART",
                "uses_additional_data": false,
                "paper": "bart-denoising-sequence-to-sequence-pre",
                "best_metric": "Rouge-L",
                "evaluated_on": "2019-10-29",
                "evaluation": "open-domain-question-answering-on-eli5",
                "benchmark_details": {
                    "id": "open-domain-question-answering-on-eli5",
                    "task": "open-domain-question-answering",
                    "dataset": "eli5",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "131de994-12ce-46e5-8183-2c0a55cea548",
                "best_rank": 7.0,
                "metrics": {
                    "ROUGE-1": "45.14",
                    "ROUGE-2": "22.27",
                    "ROUGE-3": "37.25"
                },
                "methodology": "BART",
                "uses_additional_data": false,
                "paper": "bart-denoising-sequence-to-sequence-pre",
                "best_metric": "ROUGE-1",
                "evaluated_on": "2019-10-29",
                "evaluation": "text-summarization-on-x-sum",
                "benchmark_details": {
                    "id": "text-summarization-on-x-sum",
                    "task": "text-summarization",
                    "dataset": "x-sum",
                    "description": "",
                    "mirror_url": null
                }
            }
        ],
        "model_usage": {
            "llm_input": {
                "bart-large-mnli": {
                    "bart-large-mnli/ NLI-based Zero Shot Text Classification": "Yin et al. proposed a method for using pre-trained NLI models as a ready-made zero-shot sequence classifiers. The method works by posing the sequence to be classified as the NLI premise and to construct a hypothesis from each candidate label. For example, if we want to evaluate whether a sequence belongs to the class \"politics\", we could construct a hypothesis of This text is about politics.. The probabilities for entailment and contradiction are then converted to label probabilities.\nThis method is surprisingly effective in many cases, particularly when used with larger pre-trained models like BART and Roberta. See this blog post for a more expansive introduction to this and other zero shot methods, and see the code snippets below for examples of using this model for zero-shot classification both with Hugging Face's built-in pipeline and with native Transformers/PyTorch code.\nWith the zero-shot classification pipeline\nThe model can be loaded with the zero-shot-classification pipeline like so:\nYou can then use this pipeline to classify sequences into any of the class names you specify.\nIf more than one candidate label can be correct, pass multi_label=True to calculate each class independently:\nWith manual PyTorch"
                }
            },
            "usage": {},
            "model_function": [
                {
                    "function_info": {
                        "return": "str",
                        "function_name": "text_classification_using_bart",
                        "variables": [
                            {
                                "name": "input_text",
                                "type": "str",
                                "default": "one day I will see the world"
                            },
                            {
                                "name": "labels",
                                "type": "list",
                                "default": [
                                    "travel",
                                    "cooking",
                                    "dancing"
                                ]
                            }
                        ]
                    }
                }
            ]
        }
    },
    "facebook/detr-resnet-101": {
        "model_name": "detr-resnet-101",
        "org": "facebook",
        "model_info": {
            "id": "facebook/detr-resnet-101",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 199711,
            "likes": 75,
            "library_name": "transformers",
            "tags": [
                "transformers",
                "pytorch",
                "safetensors",
                "detr",
                "object-detection",
                "vision",
                "dataset:coco",
                "arxiv:2005.12872",
                "license:apache-2.0",
                "endpoints_compatible",
                "has_space",
                "region:us"
            ],
            "pipeline_tag": "object-detection",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "621ffdc136468d709f17addf",
            "createdAt": "2022-03-02T23:29:05.000Z",
            "modelId": "facebook/detr-resnet-101"
        },
        "card_to_dict": {
            "license": "apache-2.0",
            "tags": [
                "object-detection",
                "vision"
            ],
            "datasets": [
                "coco"
            ],
            "widget": [
                {
                    "src": "https://huggingface.co/datasets/mishig/sample_images/resolve/main/savanna.jpg",
                    "example_title": "Savanna"
                },
                {
                    "src": "https://huggingface.co/datasets/mishig/sample_images/resolve/main/football-match.jpg",
                    "example_title": "Football Match"
                },
                {
                    "src": "https://huggingface.co/datasets/mishig/sample_images/resolve/main/airport.jpg",
                    "example_title": "Airport"
                }
            ]
        },
        "relevant_websites": [
            "https://huggingface.co/datasets/mishig/sample_images/resolve/main/savanna.jpg",
            "https://huggingface.co/datasets/mishig/sample_images/resolve/main/football-match.jpg",
            "https://huggingface.co/datasets/mishig/sample_images/resolve/main/airport.jpg",
            "https://arxiv.org/abs/2005.12872",
            "https://github.com/facebookresearch/detr",
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/detr_architecture.png",
            "https://huggingface.co/models?search=facebook/detr",
            "https://cocodataset.org/#download",
            "https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py",
            "https://arxiv.org/abs/2005.12872",
            "https://dblp.org/rec/journals/corr/abs-2005-12872.bib",
            "https://dblp.org"
        ],
        "text": "license: apache-2.0 tags: - object-detection - vision datasets: - coco widget: - src:    example_title: Savanna - src:    example_title: Football Match - src:    example_title: Airport  DETR (End-to-End Object Detection) model with ResNet-101 backbone DEtection TRansformer (DETR) model trained end-to-end on COCO 2017 object detection (118k annotated images). It was introduced in the paper End-to-End Object Detection with Transformers by Carion et al. and first released in this repository.  Disclaimer: The team releasing DETR did not write a model card for this model so this model card has been written by the Hugging Face team. Model description The DETR model is an encoder-decoder transformer with a convolutional backbone. Two heads are added on top of the decoder outputs in order to perform object detection: a linear layer for the class labels and a MLP (multi-layer perceptron) for the bounding boxes. The model uses so-called object queries to detect objects in an image. Each object query looks for a particular object in the image. For COCO, the number of object queries is set to 100.  The model is trained using a \"bipartite matching loss\": one compares the predicted classes + bounding boxes of each of the N = 100 object queries to the ground truth annotations, padded up to the same length N (so if an image only contains 4 objects, 96 annotations will just have a \"no object\" as class and \"no bounding box\" as bounding box). The Hungarian matching algorithm is used to create an optimal one-to-one mapping between each of the N queries and each of the N annotations. Next, standard cross-entropy (for the classes) and a linear combination of the L1 and generalized IoU loss (for the bounding boxes) are used to optimize the parameters of the model.  Intended uses & limitations You can use the raw model for object detection. See the model hub to look for all available DETR models. How to use Here is how to use this model: This should output (something along the lines of): Detected cat with confidence 0.998 at location [344.06, 24.85, 640.34, 373.74] Detected remote with confidence 0.997 at location [328.13, 75.93, 372.81, 187.66] Detected remote with confidence 0.997 at location [39.34, 70.13, 175.56, 118.78] Detected cat with confidence 0.998 at location [15.36, 51.75, 316.89, 471.16] Detected couch with confidence 0.995 at location [-0.19, 0.71, 639.73, 474.17] Currently, both the feature extractor and model support PyTorch.  Training data The DETR model was trained on COCO 2017 object detection, a dataset consisting of 118k/5k annotated images for training/validation respectively.  Training procedure Preprocessing The exact details of preprocessing of images during training/validation can be found here.  Images are resized/rescaled such that the shortest side is at least 800 pixels and the largest side at most 1333 pixels, and normalized across the RGB channels with the ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225). Training The model was trained for 300 epochs on 16 V100 GPUs. This takes 3 days, with 4 images per GPU (hence a total batch size of 64). Evaluation results This model achieves an AP (average precision) of 43.5 on COCO 2017 validation. For more details regarding evaluation results, we refer to table 1 of the original paper. BibTeX entry and citation info bibtex @article{DBLP:journals/corr/abs-2005-12872,   author    = {Nicolas Carion and                Francisco Massa and                Gabriel Synnaeve and                Nicolas Usunier and                Alexander Kirillov and                Sergey Zagoruyko},   title     = {End-to-End Object Detection with Transformers},   journal   = {CoRR},   volume    = {abs/2005.12872},   year      = {2020},   url       = {},   archivePrefix = {arXiv},   eprint    = {2005.12872},   timestamp = {Thu, 28 May 2020 17:38:09 +0200},   biburl    = {},   bibsource = {dblp computer science bibliography, } }",
        "markdown_text": "---\nlicense: apache-2.0\ntags:\n- object-detection\n- vision\ndatasets:\n- coco\nwidget:\n- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/savanna.jpg\n  example_title: Savanna\n- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/football-match.jpg\n  example_title: Football Match\n- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/airport.jpg\n  example_title: Airport\n---\n\n# DETR (End-to-End Object Detection) model with ResNet-101 backbone\n\nDEtection TRansformer (DETR) model trained end-to-end on COCO 2017 object detection (118k annotated images). It was introduced in the paper [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) by Carion et al. and first released in [this repository](https://github.com/facebookresearch/detr). \n\nDisclaimer: The team releasing DETR did not write a model card for this model so this model card has been written by the Hugging Face team.\n\n## Model description\n\nThe DETR model is an encoder-decoder transformer with a convolutional backbone. Two heads are added on top of the decoder outputs in order to perform object detection: a linear layer for the class labels and a MLP (multi-layer perceptron) for the bounding boxes. The model uses so-called object queries to detect objects in an image. Each object query looks for a particular object in the image. For COCO, the number of object queries is set to 100. \n\nThe model is trained using a \"bipartite matching loss\": one compares the predicted classes + bounding boxes of each of the N = 100 object queries to the ground truth annotations, padded up to the same length N (so if an image only contains 4 objects, 96 annotations will just have a \"no object\" as class and \"no bounding box\" as bounding box). The Hungarian matching algorithm is used to create an optimal one-to-one mapping between each of the N queries and each of the N annotations. Next, standard cross-entropy (for the classes) and a linear combination of the L1 and generalized IoU loss (for the bounding boxes) are used to optimize the parameters of the model.\n\n![model image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/detr_architecture.png)\n\n## Intended uses & limitations\n\nYou can use the raw model for object detection. See the [model hub](https://huggingface.co/models?search=facebook/detr) to look for all available DETR models.\n\n### How to use\n\nHere is how to use this model:\n\n```python\nfrom transformers import DetrImageProcessor, DetrForObjectDetection\nimport torch\nfrom PIL import Image\nimport requests\n\nurl = \"http://images.cocodataset.org/val2017/000000039769.jpg\"\nimage = Image.open(requests.get(url, stream=True).raw)\n\n# you can specify the revision tag if you don't want the timm dependency\nprocessor = DetrImageProcessor.from_pretrained(\"facebook/detr-resnet-101\", revision=\"no_timm\")\nmodel = DetrForObjectDetection.from_pretrained(\"facebook/detr-resnet-101\", revision=\"no_timm\")\n\ninputs = processor(images=image, return_tensors=\"pt\")\noutputs = model(**inputs)\n\n# convert outputs (bounding boxes and class logits) to COCO API\n# let's only keep detections with score > 0.9\ntarget_sizes = torch.tensor([image.size[::-1]])\nresults = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]\n\nfor score, label, box in zip(results[\"scores\"], results[\"labels\"], results[\"boxes\"]):\n    box = [round(i, 2) for i in box.tolist()]\n    print(\n            f\"Detected {model.config.id2label[label.item()]} with confidence \"\n            f\"{round(score.item(), 3)} at location {box}\"\n    )\n```\nThis should output (something along the lines of):\n```\nDetected cat with confidence 0.998 at location [344.06, 24.85, 640.34, 373.74]\nDetected remote with confidence 0.997 at location [328.13, 75.93, 372.81, 187.66]\nDetected remote with confidence 0.997 at location [39.34, 70.13, 175.56, 118.78]\nDetected cat with confidence 0.998 at location [15.36, 51.75, 316.89, 471.16]\nDetected couch with confidence 0.995 at location [-0.19, 0.71, 639.73, 474.17]\n```\n\nCurrently, both the feature extractor and model support PyTorch. \n\n## Training data\n\nThe DETR model was trained on [COCO 2017 object detection](https://cocodataset.org/#download), a dataset consisting of 118k/5k annotated images for training/validation respectively. \n\n## Training procedure\n\n### Preprocessing\n\nThe exact details of preprocessing of images during training/validation can be found [here](https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py). \n\nImages are resized/rescaled such that the shortest side is at least 800 pixels and the largest side at most 1333 pixels, and normalized across the RGB channels with the ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225).\n\n### Training\n\nThe model was trained for 300 epochs on 16 V100 GPUs. This takes 3 days, with 4 images per GPU (hence a total batch size of 64).\n\n## Evaluation results\n\nThis model achieves an AP (average precision) of **43.5** on COCO 2017 validation. For more details regarding evaluation results, we refer to table 1 of the original paper.\n### BibTeX entry and citation info\n\n```bibtex\n@article{DBLP:journals/corr/abs-2005-12872,\n  author    = {Nicolas Carion and\n               Francisco Massa and\n               Gabriel Synnaeve and\n               Nicolas Usunier and\n               Alexander Kirillov and\n               Sergey Zagoruyko},\n  title     = {End-to-End Object Detection with Transformers},\n  journal   = {CoRR},\n  volume    = {abs/2005.12872},\n  year      = {2020},\n  url       = {https://arxiv.org/abs/2005.12872},\n  archivePrefix = {arXiv},\n  eprint    = {2005.12872},\n  timestamp = {Thu, 28 May 2020 17:38:09 +0200},\n  biburl    = {https://dblp.org/rec/journals/corr/abs-2005-12872.bib},\n  bibsource = {dblp computer science bibliography, https://dblp.org}\n}\n```",
        "llm_extraction": {
            "model_name": "DETR",
            "model_framework": "transformers",
            "model_architecture": "encoder-decoder",
            "tasks": [
                "object-detection"
            ],
            "training_strategy": "bipartite matching loss",
            "parameters": "100 object queries",
            "vocab_size": "N/A",
            "training_data": "COCO 2017 object detection",
            "authors": [
                "Nicolas Carion",
                "Francisco Massa",
                "Gabriel Synnaeve",
                "Nicolas Usunier",
                "Alexander Kirillov",
                "Sergey Zagoruyko"
            ],
            "other": [
                "PyTorch"
            ]
        },
        "truncation": 0,
        "extraction_version": "v_1",
        "papers_with_code": "successful",
        "performance": [
            {
                "id": "299c452f-26f2-4cd4-a55a-f0503468f926",
                "best_rank": 21.0,
                "metrics": {
                    "PQ": "45.1",
                    "SQ": "79.9",
                    "RQ": "55.5",
                    "PQth": "50.5",
                    "SQth": "80.9",
                    "RQth": "61.7",
                    "PQst": "37",
                    "SQst": "78.5",
                    "RQst": "46",
                    "AP": "33"
                },
                "methodology": "DETR-R101 (ResNet-101)",
                "uses_additional_data": false,
                "paper": "end-to-end-object-detection-with-transformers",
                "best_metric": "PQ",
                "evaluated_on": "2020-05-26",
                "evaluation": "panoptic-segmentation-on-coco-minival",
                "benchmark_details": {
                    "id": "panoptic-segmentation-on-coco-minival",
                    "task": "panoptic-segmentation",
                    "dataset": "coco-minival",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "10051cd3-e929-489c-b259-8369c976b1f0",
                "best_rank": NaN,
                "metrics": {
                    "box AP": "44.9",
                    "AP50": "64.7",
                    "AP75": "47.7",
                    "APS": "23.7",
                    "APM": "49.5",
                    "APL": "62.3"
                },
                "methodology": "DETR-DC5 (ResNet-101)",
                "uses_additional_data": false,
                "paper": "end-to-end-object-detection-with-transformers",
                "best_metric": null,
                "evaluated_on": "2020-05-26",
                "evaluation": "object-detection-on-coco-minival",
                "benchmark_details": {
                    "id": "object-detection-on-coco-minival",
                    "task": "object-detection",
                    "dataset": "coco-minival",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "d9ede094-801c-44cf-9f52-ec8d01bc0f1c",
                "best_rank": NaN,
                "metrics": {
                    "Average mAP": "17.1",
                    "Effective Robustness": "-1.82"
                },
                "methodology": "DETR\n(ResNet-50)",
                "uses_additional_data": false,
                "paper": "end-to-end-object-detection-with-transformers",
                "best_metric": null,
                "evaluated_on": "2020-05-26",
                "evaluation": "object-detection-on-coco-o",
                "benchmark_details": {
                    "id": "object-detection-on-coco-o",
                    "task": "object-detection",
                    "dataset": "coco-o",
                    "description": "",
                    "mirror_url": null
                }
            }
        ],
        "model_usage": {
            "llm_input": {
                "DETR (End-to-End Object Detection) model with ResNet-101 backbone": {
                    "DETR (End-to-End Object Detection) model with ResNet-101 backbone/ Model description": "The DETR model is an encoder-decoder transformer with a convolutional backbone. Two heads are added on top of the decoder outputs in order to perform object detection: a linear layer for the class labels and a MLP (multi-layer perceptron) for the bounding boxes. The model uses so-called object queries to detect objects in an image. Each object query looks for a particular object in the image. For COCO, the number of object queries is set to 100. \nThe model is trained using a \"bipartite matching loss\": one compares the predicted classes + bounding boxes of each of the N = 100 object queries to the ground truth annotations, padded up to the same length N (so if an image only contains 4 objects, 96 annotations will just have a \"no object\" as class and \"no bounding box\" as bounding box). The Hungarian matching algorithm is used to create an optimal one-to-one mapping between each of the N queries and each of the N annotations. Next, standard cross-entropy (for the classes) and a linear combination of the L1 and generalized IoU loss (for the bounding boxes) are used to optimize the parameters of the model.\n",
                    "DETR (End-to-End Object Detection) model with ResNet-101 backbone/ Training data": "The DETR model was trained on COCO 2017 object detection, a dataset consisting of 118k/5k annotated images for training/validation respectively. ",
                    "Training procedure/ Preprocessing": "The exact details of preprocessing of images during training/validation can be found here. \nImages are resized/rescaled such that the shortest side is at least 800 pixels and the largest side at most 1333 pixels, and normalized across the RGB channels with the ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225).",
                    "Training procedure/ Training": "The model was trained for 300 epochs on 16 V100 GPUs. This takes 3 days, with 4 images per GPU (hence a total batch size of 64).",
                    "Evaluation results/ BibTeX entry and citation info": "@article{DBLP:journals/corr/abs-2005-12872,\n  author    = {Nicolas Carion and\n               Francisco Massa and\n               Gabriel Synnaeve and\n               Nicolas Usunier and\n               Alexander Kirillov and\n               Sergey Zagoruyko},\n  title     = {End-to-End Object Detection with Transformers},\n  journal   = {CoRR},\n  volume    = {abs/2005.12872},\n  year      = {2020},\n  url       = {https://arxiv.org/abs/2005.12872},\n  archivePrefix = {arXiv},\n  eprint    = {2005.12872},\n  timestamp = {Thu, 28 May 2020 17:38:09 +0200},\n  biburl    = {https://dblp.org/rec/journals/corr/abs-2005-12872.bib},\n  bibsource = {dblp computer science bibliography, https://dblp.org}\n}"
                }
            },
            "usage": {
                "Intended uses & limitations/ How to use": "Here is how to use this model:\n```\nfrom transformers import DetrImageProcessor, DetrForObjectDetection\nimport torch\nfrom PIL import Image\nimport requests\nurl = \"http://images.cocodataset.org/val2017/000000039769.jpg\"\nimage = Image.open(requests.get(url, stream=True).raw)\nyou can specify the revision tag if you don't want the timm dependency\nprocessor = DetrImageProcessor.from_pretrained(\"facebook/detr-resnet-101\", revision=\"no_timm\")\nmodel = DetrForObjectDetection.from_pretrained(\"facebook/detr-resnet-101\", revision=\"no_timm\")\ninputs = processor(images=image, return_tensors=\"pt\")\noutputs = model(**inputs)\nconvert outputs (bounding boxes and class logits) to COCO API\nlet's only keep detections with score > 0.9\ntarget_sizes = torch.tensor([image.size[::-1]])\nresults = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]\nfor score, label, box in zip(results[\"scores\"], results[\"labels\"], results[\"boxes\"]):\n    box = [round(i, 2) for i in box.tolist()]\n    print(\n            f\"Detected {model.config.id2label[label.item()]} with confidence \"\n            f\"{round(score.item(), 3)} at location {box}\"\n    )\n```\nThis should output (something along the lines of):\nDetected cat with confidence 0.998 at location [344.06, 24.85, 640.34, 373.74]\nDetected remote with confidence 0.997 at location [328.13, 75.93, 372.81, 187.66]\nDetected remote with confidence 0.997 at location [39.34, 70.13, 175.56, 118.78]\nDetected cat with confidence 0.998 at location [15.36, 51.75, 316.89, 471.16]\nDetected couch with confidence 0.995 at location [-0.19, 0.71, 639.73, 474.17]\nCurrently, both the feature extractor and model support PyTorch. "
            },
            "model_function": [
                {
                    "code": "import requests\nfrom PIL import Image\nfrom transformers import DetrImageProcessor, DetrForObjectDetection\nimport torch\n\ndef detect_objects(image_url, model_name=\"facebook/detr-resnet-101\", revision=\"no_timm\", threshold=0.9):\n    # Load the image\n    image = Image.open(requests.get(image_url, stream=True).raw)\n\n    # Load the model and processor\n    processor = DetrImageProcessor.from_pretrained(model_name, revision=revision)\n    model = DetrForObjectDetection.from_pretrained(model_name, revision=revision)\n\n    # Preprocess the image\n    inputs = processor(images=image, return_tensors=\"pt\")\n\n    # Run inference\n    outputs = model(**inputs)\n\n    # Post-process outputs\n    target_sizes = torch.tensor([image.size[::-1]])\n    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]\n\n    # Print the results\n    for score, label, box in zip(results[\"scores\"], results[\"labels\"], results[\"boxes\"]):\n        box = [round(i, 2) for i in box.tolist()]\n        print(\n            f\"Detected {model.config.id2label[label.item()]} with confidence \"\n            f\"{round(score.item(), 3)} at location {box}\"\n        )",
                    "function_info": {
                        "return": null,
                        "function_name": "object_detection",
                        "variables": [
                            {
                                "name": "input",
                                "type": "str",
                                "default": "./elephant.jpeg"
                            }
                        ]
                    }
                }
            ]
        }
    },
    "google/vit-base-patch16-224": {
        "model_name": "vit-base-patch16-224",
        "org": "google",
        "model_info": {
            "id": "google/vit-base-patch16-224",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 3515852,
            "likes": 449,
            "library_name": "transformers",
            "tags": [
                "transformers",
                "pytorch",
                "tf",
                "jax",
                "safetensors",
                "vit",
                "image-classification",
                "vision",
                "dataset:imagenet-1k",
                "dataset:imagenet-21k",
                "arxiv:2010.11929",
                "arxiv:2006.03677",
                "license:apache-2.0",
                "autotrain_compatible",
                "endpoints_compatible",
                "has_space",
                "region:us"
            ],
            "pipeline_tag": "image-classification",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "621ffdc136468d709f17b7d7",
            "createdAt": "2022-03-02T23:29:05.000Z",
            "modelId": "google/vit-base-patch16-224"
        },
        "card_to_dict": {
            "license": "apache-2.0",
            "tags": [
                "vision",
                "image-classification"
            ],
            "datasets": [
                "imagenet-1k",
                "imagenet-21k"
            ],
            "widget": [
                {
                    "src": "https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg",
                    "example_title": "Tiger"
                },
                {
                    "src": "https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg",
                    "example_title": "Teapot"
                },
                {
                    "src": "https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg",
                    "example_title": "Palace"
                }
            ]
        },
        "relevant_websites": [
            "https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg",
            "https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg",
            "https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg",
            "https://arxiv.org/abs/2010.11929",
            "https://github.com/google-research/vision_transformer",
            "https://github.com/rwightman/pytorch-image-models",
            "https://huggingface.co/models?search=google/vit",
            "https://huggingface.co/transformers/model_doc/vit.html",
            "http://www.image-net.org",
            "http://www.image-net.org/challenges/LSVRC/2012",
            "https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py"
        ],
        "text": "license: apache-2.0 tags: - vision - image-classification datasets: - imagenet-1k - imagenet-21k widget: - src:    example_title: Tiger - src:    example_title: Teapot - src:    example_title: Palace  Vision Transformer (base-sized model) Vision Transformer (ViT) model pre-trained on ImageNet-21k (14 million images, 21,843 classes) at resolution 224x224, and fine-tuned on ImageNet 2012 (1 million images, 1,000 classes) at resolution 224x224. It was introduced in the paper An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale by Dosovitskiy et al. and first released in this repository. However, the weights were converted from the timm repository by Ross Wightman, who already converted the weights from JAX to PyTorch. Credits go to him.  Disclaimer: The team releasing ViT did not write a model card for this model so this model card has been written by the Hugging Face team. Model description The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a supervised fashion, namely ImageNet-21k, at a resolution of 224x224 pixels. Next, the model was fine-tuned on ImageNet (also referred to as ILSVRC2012), a dataset comprising 1 million images and 1,000 classes, also at resolution 224x224. Images are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder. By pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled images for instance, you can train a standard classifier by placing a linear layer on top of the pre-trained encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image. Intended uses & limitations You can use the raw model for image classification. See the model hub to look for fine-tuned versions on a task that interests you. How to use Here is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes: For more code examples, we refer to the documentation. Training data The ViT model was pretrained on ImageNet-21k, a dataset consisting of 14 million images and 21k classes, and fine-tuned on ImageNet, a dataset consisting of 1 million images and 1k classes.  Training procedure Preprocessing The exact details of preprocessing of images during training/validation can be found here.  Images are resized/rescaled to the same resolution (224x224) and normalized across the RGB channels with mean (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5). Pretraining The model was trained on TPUv3 hardware (8 cores). All model variants are trained with a batch size of 4096 and learning rate warmup of 10k steps. For ImageNet, the authors found it beneficial to additionally apply gradient clipping at global norm 1. Training resolution is 224. Evaluation results For evaluation results on several image classification benchmarks, we refer to tables 2 and 5 of the original paper. Note that for fine-tuning, the best results are obtained with a higher resolution (384x384). Of course, increasing the model size will result in better performance. BibTeX entry and citation info bibtex @misc{wu2020visual,       title={Visual Transformers: Token-based Image Representation and Processing for Computer Vision},        author={Bichen Wu and Chenfeng Xu and Xiaoliang Dai and Alvin Wan and Peizhao Zhang and Zhicheng Yan and Masayoshi Tomizuka and Joseph Gonzalez and Kurt Keutzer and Peter Vajda},       year={2020},       eprint={2006.03677},       archivePrefix={arXiv},       primaryClass={cs.CV} } bibtex @inproceedings{deng2009imagenet,   title={Imagenet: A large-scale hierarchical image database},   author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},   booktitle={2009 IEEE conference on computer vision and pattern recognition},   pages={248--255},   year={2009},   organization={Ieee} }",
        "markdown_text": "---\nlicense: apache-2.0\ntags:\n- vision\n- image-classification\ndatasets:\n- imagenet-1k\n- imagenet-21k\nwidget:\n- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg\n  example_title: Tiger\n- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg\n  example_title: Teapot\n- src: https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg\n  example_title: Palace\n---\n\n# Vision Transformer (base-sized model) \n\nVision Transformer (ViT) model pre-trained on ImageNet-21k (14 million images, 21,843 classes) at resolution 224x224, and fine-tuned on ImageNet 2012 (1 million images, 1,000 classes) at resolution 224x224. It was introduced in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al. and first released in [this repository](https://github.com/google-research/vision_transformer). However, the weights were converted from the [timm repository](https://github.com/rwightman/pytorch-image-models) by Ross Wightman, who already converted the weights from JAX to PyTorch. Credits go to him. \n\nDisclaimer: The team releasing ViT did not write a model card for this model so this model card has been written by the Hugging Face team.\n\n## Model description\n\nThe Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a supervised fashion, namely ImageNet-21k, at a resolution of 224x224 pixels. Next, the model was fine-tuned on ImageNet (also referred to as ILSVRC2012), a dataset comprising 1 million images and 1,000 classes, also at resolution 224x224.\n\nImages are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder.\n\nBy pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled images for instance, you can train a standard classifier by placing a linear layer on top of the pre-trained encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.\n\n## Intended uses & limitations\n\nYou can use the raw model for image classification. See the [model hub](https://huggingface.co/models?search=google/vit) to look for\nfine-tuned versions on a task that interests you.\n\n### How to use\n\nHere is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes:\n\n```python\nfrom transformers import ViTImageProcessor, ViTForImageClassification\nfrom PIL import Image\nimport requests\n\nurl = 'http://images.cocodataset.org/val2017/000000039769.jpg'\nimage = Image.open(requests.get(url, stream=True).raw)\n\nprocessor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')\nmodel = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')\n\ninputs = processor(images=image, return_tensors=\"pt\")\noutputs = model(**inputs)\nlogits = outputs.logits\n# model predicts one of the 1000 ImageNet classes\npredicted_class_idx = logits.argmax(-1).item()\nprint(\"Predicted class:\", model.config.id2label[predicted_class_idx])\n```\n\nFor more code examples, we refer to the [documentation](https://huggingface.co/transformers/model_doc/vit.html#).\n\n## Training data\n\nThe ViT model was pretrained on [ImageNet-21k](http://www.image-net.org/), a dataset consisting of 14 million images and 21k classes, and fine-tuned on [ImageNet](http://www.image-net.org/challenges/LSVRC/2012/), a dataset consisting of 1 million images and 1k classes. \n\n## Training procedure\n\n### Preprocessing\n\nThe exact details of preprocessing of images during training/validation can be found [here](https://github.com/google-research/vision_transformer/blob/master/vit_jax/input_pipeline.py). \n\nImages are resized/rescaled to the same resolution (224x224) and normalized across the RGB channels with mean (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5).\n\n### Pretraining\n\nThe model was trained on TPUv3 hardware (8 cores). All model variants are trained with a batch size of 4096 and learning rate warmup of 10k steps. For ImageNet, the authors found it beneficial to additionally apply gradient clipping at global norm 1. Training resolution is 224.\n\n## Evaluation results\n\nFor evaluation results on several image classification benchmarks, we refer to tables 2 and 5 of the original paper. Note that for fine-tuning, the best results are obtained with a higher resolution (384x384). Of course, increasing the model size will result in better performance.\n\n### BibTeX entry and citation info\n\n```bibtex\n@misc{wu2020visual,\n      title={Visual Transformers: Token-based Image Representation and Processing for Computer Vision}, \n      author={Bichen Wu and Chenfeng Xu and Xiaoliang Dai and Alvin Wan and Peizhao Zhang and Zhicheng Yan and Masayoshi Tomizuka and Joseph Gonzalez and Kurt Keutzer and Peter Vajda},\n      year={2020},\n      eprint={2006.03677},\n      archivePrefix={arXiv},\n      primaryClass={cs.CV}\n}\n```\n\n```bibtex\n@inproceedings{deng2009imagenet,\n  title={Imagenet: A large-scale hierarchical image database},\n  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},\n  booktitle={2009 IEEE conference on computer vision and pattern recognition},\n  pages={248--255},\n  year={2009},\n  organization={Ieee}\n}\n```",
        "llm_extraction": {
            "model_name": "ViT",
            "model_framework": "transformers",
            "model_architecture": "BERT-like",
            "tasks": [
                "image-classification"
            ],
            "training_strategy": "supervised",
            "parameters": "14 million images, 21k classes, 1 million images, 1k classes",
            "vocab_size": "NONE",
            "training_data": "ImageNet-21k, ImageNet",
            "authors": [
                "NONE"
            ],
            "other": [
                "CLS token",
                "absolute position embeddings",
                "PyTorch",
                "TPUv3 hardware",
                "batch size",
                "learning rate warmup",
                "gradient clipping",
                "resolution",
                "evaluation results"
            ]
        },
        "truncation": 0,
        "extraction_version": "v_1",
        "papers_with_code": "successful",
        "performance": [
            {
                "id": "a6b68ddf-9227-45a0-9140-1f779cc77720",
                "best_rank": NaN,
                "metrics": {
                    "WAR": "45.04"
                },
                "methodology": "ViT",
                "uses_additional_data": false,
                "paper": "an-image-is-worth-16x16-words-transformers-1",
                "best_metric": null,
                "evaluated_on": "2020-10-22",
                "evaluation": "dynamic-facial-expression-recognition-on-mafw",
                "benchmark_details": {
                    "id": "dynamic-facial-expression-recognition-on-mafw",
                    "task": "dynamic-facial-expression-recognition",
                    "dataset": "mafw",
                    "description": "",
                    "mirror_url": null
                }
            }
        ],
        "model_usage": {
            "llm_input": {
                "Vision Transformer (base-sized model) ": {
                    "Vision Transformer (base-sized model) / Model description": "The Vision Transformer (ViT) is a transformer encoder model (BERT-like) pretrained on a large collection of images in a supervised fashion, namely ImageNet-21k, at a resolution of 224x224 pixels. Next, the model was fine-tuned on ImageNet (also referred to as ILSVRC2012), a dataset comprising 1 million images and 1,000 classes, also at resolution 224x224.\nImages are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded. One also adds a [CLS] token to the beginning of a sequence to use it for classification tasks. One also adds absolute position embeddings before feeding the sequence to the layers of the Transformer encoder.\nBy pre-training the model, it learns an inner representation of images that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled images for instance, you can train a standard classifier by placing a linear layer on top of the pre-trained encoder. One typically places a linear layer on top of the [CLS] token, as the last hidden state of this token can be seen as a representation of an entire image.",
                    "Vision Transformer (base-sized model) / Training data": "The ViT model was pretrained on ImageNet-21k, a dataset consisting of 14 million images and 21k classes, and fine-tuned on ImageNet, a dataset consisting of 1 million images and 1k classes. ",
                    "Training procedure/ Preprocessing": "The exact details of preprocessing of images during training/validation can be found here. \nImages are resized/rescaled to the same resolution (224x224) and normalized across the RGB channels with mean (0.5, 0.5, 0.5) and standard deviation (0.5, 0.5, 0.5).",
                    "Training procedure/ Pretraining": "The model was trained on TPUv3 hardware (8 cores). All model variants are trained with a batch size of 4096 and learning rate warmup of 10k steps. For ImageNet, the authors found it beneficial to additionally apply gradient clipping at global norm 1. Training resolution is 224.",
                    "Evaluation results/ BibTeX entry and citation info": "@misc{wu2020visual,\n      title={Visual Transformers: Token-based Image Representation and Processing for Computer Vision}, \n      author={Bichen Wu and Chenfeng Xu and Xiaoliang Dai and Alvin Wan and Peizhao Zhang and Zhicheng Yan and Masayoshi Tomizuka and Joseph Gonzalez and Kurt Keutzer and Peter Vajda},\n      year={2020},\n      eprint={2006.03677},\n      archivePrefix={arXiv},\n      primaryClass={cs.CV}\n}\n@inproceedings{deng2009imagenet,\n  title={Imagenet: A large-scale hierarchical image database},\n  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},\n  booktitle={2009 IEEE conference on computer vision and pattern recognition},\n  pages={248--255},\n  year={2009},\n  organization={Ieee}\n}"
                }
            },
            "usage": {
                "Intended uses & limitations/ How to use": "Here is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes:\n```\nfrom transformers import ViTImageProcessor, ViTForImageClassification\nfrom PIL import Image\nimport requests\nurl = 'http://images.cocodataset.org/val2017/000000039769.jpg'\nimage = Image.open(requests.get(url, stream=True).raw)\nprocessor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')\nmodel = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')\ninputs = processor(images=image, return_tensors=\"pt\")\noutputs = model(**inputs)\nlogits = outputs.logits\nmodel predicts one of the 1000 ImageNet classes\npredicted_class_idx = logits.argmax(-1).item()\nprint(\"Predicted class:\", model.config.id2label[predicted_class_idx])\n```\nFor more code examples, we refer to the documentation."
            },
            "model_function": [
                {
                    "code": "from transformers import ViTImageProcessor, ViTForImageClassification\nfrom PIL import Image\nimport requests\n\ndef classify_image(url='http://images.cocodataset.org/val2017/000000039769.jpg',\n                   model_name='google/vit-base-patch16-224'):\n    image = Image.open(requests.get(url, stream=True).raw)\n    processor = ViTImageProcessor.from_pretrained(model_name)\n    model = ViTForImageClassification.from_pretrained(model_name)\n    inputs = processor(images=image, return_tensors=\"pt\")\n    outputs = model(**inputs)\n    logits = outputs.logits\n    predicted_class_idx = logits.argmax(-1).item()\n    print(\"Predicted class:\", model.config.id2label[predicted_class_idx])",
                    "function_info": {
                        "return": null,
                        "function_name": "classify_image",
                        "variables": [
                            {
                                "name": "input",
                                "type": "str",
                                "default": "./elephant.jpeg"
                            }
                        ]
                    }
                }
            ]
        }
    },
    "microsoft/speecht5_tts": {
        "model_name": "speecht5_tts",
        "org": "microsoft",
        "model_info": {
            "id": "microsoft/speecht5_tts",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 533363,
            "likes": 437,
            "library_name": "transformers",
            "tags": [
                "transformers",
                "pytorch",
                "speecht5",
                "text-to-audio",
                "audio",
                "text-to-speech",
                "dataset:libritts",
                "arxiv:2110.07205",
                "arxiv:1910.09700",
                "license:mit",
                "endpoints_compatible",
                "has_space",
                "region:us"
            ],
            "pipeline_tag": "text-to-audio",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "63dbb316057a688a88b910e5",
            "createdAt": "2023-02-02T12:56:54.000Z",
            "modelId": "microsoft/speecht5_tts"
        },
        "card_to_dict": {
            "license": "mit",
            "tags": [
                "audio",
                "text-to-speech"
            ],
            "datasets": [
                "libritts"
            ]
        },
        "relevant_websites": [
            "https://arxiv.org/abs/2110.07205",
            "https://github.com/microsoft/SpeechT5",
            "https://huggingface.co/mechanicalsea/speecht5-tts",
            "https://github.com/microsoft/SpeechT5/blob/main/LICENSE",
            "https://huggingface.co/Matthijs",
            "https://github.com/microsoft/SpeechT5/blob/main/LICENSE",
            "https://github.com/microsoft/SpeechT5",
            "https://arxiv.org/pdf/2110.07205.pdf",
            "https://huggingface.co/blog/speecht5",
            "https://huggingface.co/spaces/Matthijs/speecht5-tts-demo",
            "https://github.com/huggingface/transformers",
            "https://colab.research.google.com/drive/1i7I5pzBcU3WDFarDnzweIj4-sVVoIUFJ",
            "https://huggingface.co/models?search=speecht5",
            "https://mlco2.github.io/impact#compute",
            "https://arxiv.org/abs/1910.09700"
        ],
        "text": "license: mit tags: - audio - text-to-speech datasets: - libritts  SpeechT5 (TTS task) SpeechT5 model fine-tuned for speech synthesis (text-to-speech) on LibriTTS. This model was introduced in SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing by Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei. SpeechT5 was first released in this repository, original weights. The license used is MIT. Model Description Motivated by the success of T5 (Text-To-Text Transfer Transformer) in pre-trained natural language processing models, we propose a unified-modal SpeechT5 framework that explores the encoder-decoder pre-training for self-supervised speech/text representation learning. The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder. Leveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text. To align the textual and speech information into this unified semantic space, we propose a cross-modal vector quantization approach that randomly mixes up speech/text states with latent units as the interface between encoder and decoder. Extensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.  Developed by: Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei. Shared by [optional]: Matthijs Hollemans Model type: text-to-speech Language(s) (NLP): [More Information Needed] License: MIT Finetuned from model [optional]: [More Information Needed]  Model Sources [optional]   Repository: [/] Paper: [] Blog Post: [] Demo: []  Uses  🤗 Transformers Usage You can run SpeechT5 TTS locally with the 🤗 Transformers library.  First install the 🤗 Transformers library, sentencepiece, soundfile and datasets(optional):  pip install --upgrade pip pip install --upgrade transformers sentencepiece datasets[audio]   Run inference via the Text-to-Speech (TTS) pipeline. You can access the SpeechT5 model via the TTS pipeline in just a few lines of code!   Run inference via the Transformers modelling code - You can use the processor + generate code to convert text into a mono 16 kHz speech waveform for more fine-grained control.   Fine-tuning the Model Refer to this Colab notebook for an example of how to fine-tune SpeechT5 for TTS on a different dataset or a new language. Direct Use  You can use this model for speech synthesis. See the model hub to look for fine-tuned versions on a task that interests you. Downstream Use [optional]  [More Information Needed] Out-of-Scope Use  [More Information Needed] Bias, Risks, and Limitations  [More Information Needed] Recommendations  Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations. Training Details Training Data  LibriTTS Training Procedure  Preprocessing [optional] Leveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text. Training hyperparameters  Precision: [More Information Needed]  Regime: [More Information Needed]   Speeds, Sizes, Times [optional]  [More Information Needed] Evaluation  Testing Data, Factors & Metrics Testing Data  [More Information Needed] Factors  [More Information Needed] Metrics  [More Information Needed] Results [More Information Needed] Summary Model Examination [optional]  Extensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification. Environmental Impact  Carbon emissions can be estimated using the Machine Learning Impact calculator presented in Lacoste et al. (2019).  Hardware Type: [More Information Needed] Hours used: [More Information Needed] Cloud Provider: [More Information Needed] Compute Region: [More Information Needed] Carbon Emitted: [More Information Needed]  Technical Specifications [optional] Model Architecture and Objective The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder. Compute Infrastructure [More Information Needed] Hardware [More Information Needed] Software [More Information Needed] Citation [optional]  BibTeX: bibtex @inproceedings{ao-etal-2022-speecht5,     title = {{S}peech{T}5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing},     author = {Ao, Junyi and Wang, Rui and Zhou, Long and Wang, Chengyi and Ren, Shuo and Wu, Yu and Liu, Shujie and Ko, Tom and Li, Qing and Zhang, Yu and Wei, Zhihua and Qian, Yao and Li, Jinyu and Wei, Furu},     booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},     month = {May},     year = {2022},     pages={5723--5738}, } Glossary [optional]   text-to-speech to synthesize audio  More Information [optional] [More Information Needed] Model Card Authors [optional] Disclaimer: The team releasing SpeechT5 did not write a model card for this model so this model card has been written by the Hugging Face team. Model Card Contact [More Information Needed]",
        "markdown_text": "---\nlicense: mit\ntags:\n- audio\n- text-to-speech\ndatasets:\n- libritts\n---\n\n# SpeechT5 (TTS task)\n\nSpeechT5 model fine-tuned for speech synthesis (text-to-speech) on LibriTTS.\n\nThis model was introduced in [SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://arxiv.org/abs/2110.07205) by Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei.\n\nSpeechT5 was first released in [this repository](https://github.com/microsoft/SpeechT5/), [original weights](https://huggingface.co/mechanicalsea/speecht5-tts). The license used is [MIT](https://github.com/microsoft/SpeechT5/blob/main/LICENSE).\n\n\n\n## Model Description\n\nMotivated by the success of T5 (Text-To-Text Transfer Transformer) in pre-trained natural language processing models, we propose a unified-modal SpeechT5 framework that explores the encoder-decoder pre-training for self-supervised speech/text representation learning. The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder.\n\nLeveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text. To align the textual and speech information into this unified semantic space, we propose a cross-modal vector quantization approach that randomly mixes up speech/text states with latent units as the interface between encoder and decoder.\n\nExtensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.\n\n- **Developed by:** Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei.\n- **Shared by [optional]:** [Matthijs Hollemans](https://huggingface.co/Matthijs)\n- **Model type:** text-to-speech\n- **Language(s) (NLP):** [More Information Needed]\n- **License:** [MIT](https://github.com/microsoft/SpeechT5/blob/main/LICENSE)\n- **Finetuned from model [optional]:** [More Information Needed]\n\n\n## Model Sources [optional]\n\n<!-- Provide the basic links for the model. -->\n\n- **Repository:** [https://github.com/microsoft/SpeechT5/]\n- **Paper:** [https://arxiv.org/pdf/2110.07205.pdf]\n- **Blog Post:** [https://huggingface.co/blog/speecht5]\n- **Demo:** [https://huggingface.co/spaces/Matthijs/speecht5-tts-demo]\n\n\n# Uses\n\n<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->\n\n## 🤗 Transformers Usage\n\nYou can run SpeechT5 TTS locally with the 🤗 Transformers library.\n\n1. First install the 🤗 [Transformers library](https://github.com/huggingface/transformers), sentencepiece, soundfile and datasets(optional):\n\n```\npip install --upgrade pip\npip install --upgrade transformers sentencepiece datasets[audio]\n```\n\n2. Run inference via the `Text-to-Speech` (TTS) pipeline. You can access the SpeechT5 model via the TTS pipeline in just a few lines of code!\n\n```python\nfrom transformers import pipeline\nfrom datasets import load_dataset\nimport soundfile as sf\n\nsynthesiser = pipeline(\"text-to-speech\", \"microsoft/speecht5_tts\")\n\nembeddings_dataset = load_dataset(\"Matthijs/cmu-arctic-xvectors\", split=\"validation\")\nspeaker_embedding = torch.tensor(embeddings_dataset[7306][\"xvector\"]).unsqueeze(0)\n# You can replace this embedding with your own as well.\n\nspeech = synthesiser(\"Hello, my dog is cooler than you!\", forward_params={\"speaker_embeddings\": speaker_embedding})\n\nsf.write(\"speech.wav\", speech[\"audio\"], samplerate=speech[\"sampling_rate\"])\n```\n\n3. Run inference via the Transformers modelling code - You can use the processor + generate code to convert text into a mono 16 kHz speech waveform for more fine-grained control.\n\n```python\nfrom transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan\nfrom datasets import load_dataset\nimport torch\nimport soundfile as sf\nfrom datasets import load_dataset\n\nprocessor = SpeechT5Processor.from_pretrained(\"microsoft/speecht5_tts\")\nmodel = SpeechT5ForTextToSpeech.from_pretrained(\"microsoft/speecht5_tts\")\nvocoder = SpeechT5HifiGan.from_pretrained(\"microsoft/speecht5_hifigan\")\n\ninputs = processor(text=\"Hello, my dog is cute.\", return_tensors=\"pt\")\n\n# load xvector containing speaker's voice characteristics from a dataset\nembeddings_dataset = load_dataset(\"Matthijs/cmu-arctic-xvectors\", split=\"validation\")\nspeaker_embeddings = torch.tensor(embeddings_dataset[7306][\"xvector\"]).unsqueeze(0)\n\nspeech = model.generate_speech(inputs[\"input_ids\"], speaker_embeddings, vocoder=vocoder)\n\nsf.write(\"speech.wav\", speech.numpy(), samplerate=16000)\n```\n\n### Fine-tuning the Model\n\nRefer to [this Colab notebook](https://colab.research.google.com/drive/1i7I5pzBcU3WDFarDnzweIj4-sVVoIUFJ) for an example of how to fine-tune SpeechT5 for TTS on a different dataset or a new language.\n\n\n## Direct Use\n\n<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->\n\nYou can use this model for speech synthesis. See the [model hub](https://huggingface.co/models?search=speecht5) to look for fine-tuned versions on a task that interests you.\n\n## Downstream Use [optional]\n\n<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->\n\n[More Information Needed]\n\n## Out-of-Scope Use\n\n<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->\n\n[More Information Needed]\n\n# Bias, Risks, and Limitations\n\n<!-- This section is meant to convey both technical and sociotechnical limitations. -->\n\n[More Information Needed]\n\n## Recommendations\n\n<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->\n\nUsers (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.\n\n# Training Details\n\n## Training Data\n\n<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->\n\nLibriTTS\n\n## Training Procedure \n\n<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->\n\n### Preprocessing [optional]\n\nLeveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text.\n\n\n### Training hyperparameters\n- **Precision:** [More Information Needed] <!--fp16, bf16, fp8, fp32 -->\n- **Regime:** [More Information Needed] <!--mixed precision or not -->\n\n### Speeds, Sizes, Times [optional]\n\n<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->\n\n[More Information Needed]\n\n# Evaluation\n\n<!-- This section describes the evaluation protocols and provides the results. -->\n\n## Testing Data, Factors & Metrics\n\n### Testing Data\n\n<!-- This should link to a Data Card if possible. -->\n\n[More Information Needed]\n\n### Factors\n\n<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->\n\n[More Information Needed]\n\n### Metrics\n\n<!-- These are the evaluation metrics being used, ideally with a description of why. -->\n\n[More Information Needed]\n\n## Results\n\n[More Information Needed]\n\n### Summary\n\n\n\n# Model Examination [optional]\n\n<!-- Relevant interpretability work for the model goes here -->\n\nExtensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.\n\n# Environmental Impact\n\n<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->\n\nCarbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).\n\n- **Hardware Type:** [More Information Needed]\n- **Hours used:** [More Information Needed]\n- **Cloud Provider:** [More Information Needed]\n- **Compute Region:** [More Information Needed]\n- **Carbon Emitted:** [More Information Needed]\n\n# Technical Specifications [optional]\n\n## Model Architecture and Objective\n\nThe SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets.\n\nAfter preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder.\n\n## Compute Infrastructure\n\n[More Information Needed]\n\n### Hardware\n\n[More Information Needed]\n\n### Software\n\n[More Information Needed]\n\n# Citation [optional]\n\n<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->\n\n**BibTeX:**\n\n```bibtex\n@inproceedings{ao-etal-2022-speecht5,\n    title = {{S}peech{T}5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing},\n    author = {Ao, Junyi and Wang, Rui and Zhou, Long and Wang, Chengyi and Ren, Shuo and Wu, Yu and Liu, Shujie and Ko, Tom and Li, Qing and Zhang, Yu and Wei, Zhihua and Qian, Yao and Li, Jinyu and Wei, Furu},\n    booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},\n    month = {May},\n    year = {2022},\n    pages={5723--5738},\n}\n```\n\n# Glossary [optional]\n\n<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->\n\n- **text-to-speech** to synthesize audio\n\n# More Information [optional]\n\n[More Information Needed]\n\n# Model Card Authors [optional]\n\nDisclaimer: The team releasing SpeechT5 did not write a model card for this model so this model card has been written by the Hugging Face team.\n\n# Model Card Contact\n\n[More Information Needed]\n\n\n\n",
        "llm_extraction": {
            "model_name": "SpeechT5",
            "model_framework": "transformers",
            "model_architecture": "encoder-decoder",
            "tasks": [
                "text-to-speech"
            ],
            "training_strategy": "unsupervised speech/text representation learning",
            "parameters": "124M",
            "vocab_size": "50,257",
            "training_data": "LibriTTS",
            "authors": [
                "Junyi Ao",
                "Rui Wang",
                "Long Zhou",
                "Chengyi Wang",
                "Shuo Ren",
                "Yu Wu",
                "Shujie Liu",
                "Tom Ko",
                "Qing Li",
                "Yu Zhang",
                "Zhihua Wei",
                "Yao Qian",
                "Jinyu Li",
                "Furu Wei"
            ],
            "other": [
                "Byte Pair Encoding",
                "MIT",
                "SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing",
                "LibriTTS",
                "SpeechT5 framework",
                "shared encoder-decoder network",
                "six modal-specific (speech/text) pre/post-nets",
                "cross-modal vector quantization approach",
                "extensive evaluations",
                "superiority of the proposed SpeechT5 framework",
                "automatic speech recognition",
                "speech synthesis",
                "speech translation",
                "voice conversion",
                "speech enhancement",
                "speaker identification",
                "preprocessing",
                "shared encoder-decoder network",
                "post-nets",
                "sequence-to-sequence transformation",
                "unified-modal representation",
                "carbon emissions",
                "hardware",
                "software",
                "citation",
                "BibTeX",
                "glossary",
                "model card authors",
                "model card contact"
            ]
        },
        "truncation": 0,
        "extraction_version": "v_1",
        "papers_with_code": "Performance not found on papers with code",
        "model_usage": {
            "llm_input": {
                "SpeechT5 (TTS task)/ Model Description": "Motivated by the success of T5 (Text-To-Text Transfer Transformer) in pre-trained natural language processing models, we propose a unified-modal SpeechT5 framework that explores the encoder-decoder pre-training for self-supervised speech/text representation learning. The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets. After preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder.\nLeveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text. To align the textual and speech information into this unified semantic space, we propose a cross-modal vector quantization approach that randomly mixes up speech/text states with latent units as the interface between encoder and decoder.\nExtensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.\n['Developed by: Junyi Ao, Rui Wang, Long Zhou, Chengyi Wang, Shuo Ren, Yu Wu, Shujie Liu, Tom Ko, Qing Li, Yu Zhang, Zhihua Wei, Yao Qian, Jinyu Li, Furu Wei.', 'Shared by [optional]: Matthijs Hollemans', 'Model type: text-to-speech', 'Language(s) (NLP): [More Information Needed]', 'License: MIT', 'Finetuned from model [optional]: [More Information Needed]']",
                "SpeechT5 (TTS task)/ Model Sources [optional]": "\n['Repository: [https://github.com/microsoft/SpeechT5/]', 'Paper: [https://arxiv.org/pdf/2110.07205.pdf]', 'Blog Post: [https://huggingface.co/blog/speecht5]', 'Demo: [https://huggingface.co/spaces/Matthijs/speecht5-tts-demo]']",
                "🤗 Transformers Usage/ Fine-tuning the Model": "Refer to this Colab notebook for an example of how to fine-tune SpeechT5 for TTS on a different dataset or a new language.",
                "Bias, Risks, and Limitations/ Recommendations": "\nUsers (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.",
                "Training Details/ Training Data": "\nLibriTTS",
                "Training Procedure / Preprocessing [optional]": "Leveraging large-scale unlabeled speech and text data, we pre-train SpeechT5 to learn a unified-modal representation, hoping to improve the modeling capability for both speech and text.",
                "Training Procedure / Training hyperparameters": [
                    "**Precision:** [More Information Needed] <!--fp16, bf16, fp8, fp32 -->",
                    "**Regime:** [More Information Needed] <!--mixed precision or not -->"
                ],
                "Training Procedure / Speeds, Sizes, Times [optional]": "\n[More Information Needed]",
                "Testing Data, Factors & Metrics/ Testing Data": "\n[More Information Needed]",
                "Testing Data, Factors & Metrics/ Factors": "\n[More Information Needed]",
                "Testing Data, Factors & Metrics/ Metrics": "\n[More Information Needed]",
                "Results/ Summary": "",
                "Model Examination [optional]": "\nExtensive evaluations show the superiority of the proposed SpeechT5 framework on a wide variety of spoken language processing tasks, including automatic speech recognition, speech synthesis, speech translation, voice conversion, speech enhancement, and speaker identification.",
                "Environmental Impact": "\nCarbon emissions can be estimated using the Machine Learning Impact calculator presented in Lacoste et al. (2019).\n['Hardware Type: [More Information Needed]', 'Hours used: [More Information Needed]', 'Cloud Provider: [More Information Needed]', 'Compute Region: [More Information Needed]', 'Carbon Emitted: [More Information Needed]']",
                "Technical Specifications [optional]/ Model Architecture and Objective": "The SpeechT5 framework consists of a shared encoder-decoder network and six modal-specific (speech/text) pre/post-nets.\nAfter preprocessing the input speech/text through the pre-nets, the shared encoder-decoder network models the sequence-to-sequence transformation, and then the post-nets generate the output in the speech/text modality based on the output of the decoder.",
                "Compute Infrastructure/ Hardware": "[More Information Needed]",
                "Compute Infrastructure/ Software": "[More Information Needed]",
                "Citation [optional]": "\nBibTeX:\n@inproceedings{ao-etal-2022-speecht5,\n    title = {{S}peech{T}5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing},\n    author = {Ao, Junyi and Wang, Rui and Zhou, Long and Wang, Chengyi and Ren, Shuo and Wu, Yu and Liu, Shujie and Ko, Tom and Li, Qing and Zhang, Yu and Wei, Zhihua and Qian, Yao and Li, Jinyu and Wei, Furu},\n    booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},\n    month = {May},\n    year = {2022},\n    pages={5723--5738},\n}",
                "Glossary [optional]": "\n['text-to-speech to synthesize audio']",
                "More Information [optional]": "[More Information Needed]",
                "Model Card Authors [optional]": "Disclaimer: The team releasing SpeechT5 did not write a model card for this model so this model card has been written by the Hugging Face team.",
                "Model Card Contact": "[More Information Needed]"
            },
            "usage": {
                "Uses/ Direct Use": "\nYou can use this model for speech synthesis. See the model hub to look for fine-tuned versions on a task that interests you.",
                "Uses/ Downstream Use [optional]": "\n[More Information Needed]",
                "Uses/ Out-of-Scope Use": "\n[More Information Needed]"
            },
            "model_function": [
                {
                    "function_info": {
                        "return": "str",
                        "function_name": "generate_speech",
                        "variables": [
                            {
                                "name": "text",
                                "type": "str",
                                "default": "one day I will see the world"
                            }
                        ]
                    }
                }
            ]
        }
    },
    "mistralai/Mistral-7B-Instruct-v0.1": {
        "model_name": "Mistral-7B-Instruct-v0.1",
        "org": "mistralai",
        "model_info": {
            "id": "mistralai/Mistral-7B-Instruct-v0.1",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 555451,
            "likes": 1313,
            "library_name": "transformers",
            "tags": [
                "transformers",
                "pytorch",
                "safetensors",
                "mistral",
                "text-generation",
                "finetuned",
                "conversational",
                "arxiv:2310.06825",
                "license:apache-2.0",
                "autotrain_compatible",
                "has_space",
                "text-generation-inference",
                "region:us"
            ],
            "pipeline_tag": "text-generation",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "65143cd8e31c0e2e3df713e5",
            "createdAt": "2023-09-27T14:31:52.000Z",
            "modelId": "mistralai/Mistral-7B-Instruct-v0.1"
        },
        "card_to_dict": {
            "license": "apache-2.0",
            "tags": [
                "finetuned"
            ],
            "pipeline_tag": "text-generation",
            "inference": {
                "parameters": {
                    "temperature": 0.7
                }
            }
        },
        "relevant_websites": [
            "https://huggingface.co/mistralai/Mistral-7B-v0.1",
            "https://arxiv.org/abs/2310.06825",
            "https://mistral.ai/news/announcing-mistral-7b",
            "https://huggingface.co/docs/transformers/main/chat_templating",
            "https://github.com/huggingface/transformers"
        ],
        "text": "license: apache-2.0 tags: - finetuned pipeline_tag: text-generation inference:   parameters:     temperature: 0.7  Model Card for Mistral-7B-Instruct-v0.1 The Mistral-7B-Instruct-v0.1 Large Language Model (LLM) is a instruct fine-tuned version of the Mistral-7B-v0.1 generative text model using a variety of publicly available conversation datasets. For full details of this model please read our paper and release blog post. Instruction format In order to leverage instruction fine-tuning, your prompt should be surrounded by [INST] and [/INST] tokens. The very first instruction should begin with a begin of sentence id. The next instructions should not. The assistant generation will be ended by the end-of-sentence token id. E.g. text = \"<s>[INST] What is your favourite condiment? [/INST]\" \"Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> \" \"[INST] Do you have mayonnaise recipes? [/INST]\" This format is available as a chat template via the apply_chat_template() method: Model Architecture This instruction model is based on Mistral-7B-v0.1, a transformer model with the following architecture choices: - Grouped-Query Attention - Sliding-Window Attention - Byte-fallback BPE tokenizer Troubleshooting  If you see the following error: Traceback (most recent call last): File \"\", line 1, in File \"/transformers/models/auto/auto_factory.py\", line 482, in from_pretrained config, kwargs = AutoConfig.from_pretrained( File \"/transformers/models/auto/configuration_auto.py\", line 1022, in from_pretrained config_class = CONFIG_MAPPING[config_dict[\"model_type\"]] File \"/transformers/models/auto/configuration_auto.py\", line 723, in getitem raise KeyError(key) KeyError: 'mistral'  Installing transformers from source should solve the issue pip install git+ This should not be required after transformers-v4.33.4. Limitations The Mistral 7B Instruct model is a quick demonstration that the base model can be easily fine-tuned to achieve compelling performance.  It does not have any moderation mechanisms. We're looking forward to engaging with the community on ways to make the model finely respect guardrails, allowing for deployment in environments requiring moderated outputs. The Mistral AI Team Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.",
        "markdown_text": "---\nlicense: apache-2.0\ntags:\n- finetuned\npipeline_tag: text-generation\ninference:\n  parameters:\n    temperature: 0.7\n---\n\n# Model Card for Mistral-7B-Instruct-v0.1\n\nThe Mistral-7B-Instruct-v0.1 Large Language Model (LLM) is a instruct fine-tuned version of the [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) generative text model using a variety of publicly available conversation datasets.\n\nFor full details of this model please read our [paper](https://arxiv.org/abs/2310.06825) and [release blog post](https://mistral.ai/news/announcing-mistral-7b/).\n\n## Instruction format\n\nIn order to leverage instruction fine-tuning, your prompt should be surrounded by `[INST]` and `[/INST]` tokens. The very first instruction should begin with a begin of sentence id. The next instructions should not. The assistant generation will be ended by the end-of-sentence token id.\n\nE.g.\n```\ntext = \"<s>[INST] What is your favourite condiment? [/INST]\"\n\"Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> \"\n\"[INST] Do you have mayonnaise recipes? [/INST]\"\n```\n\nThis format is available as a [chat template](https://huggingface.co/docs/transformers/main/chat_templating) via the `apply_chat_template()` method:\n\n```python\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n\ndevice = \"cuda\" # the device to load the model onto\n\nmodel = AutoModelForCausalLM.from_pretrained(\"mistralai/Mistral-7B-Instruct-v0.1\")\ntokenizer = AutoTokenizer.from_pretrained(\"mistralai/Mistral-7B-Instruct-v0.1\")\n\nmessages = [\n    {\"role\": \"user\", \"content\": \"What is your favourite condiment?\"},\n    {\"role\": \"assistant\", \"content\": \"Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!\"},\n    {\"role\": \"user\", \"content\": \"Do you have mayonnaise recipes?\"}\n]\n\nencodeds = tokenizer.apply_chat_template(messages, return_tensors=\"pt\")\n\nmodel_inputs = encodeds.to(device)\nmodel.to(device)\n\ngenerated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)\ndecoded = tokenizer.batch_decode(generated_ids)\nprint(decoded[0])\n```\n\n## Model Architecture\nThis instruction model is based on Mistral-7B-v0.1, a transformer model with the following architecture choices:\n- Grouped-Query Attention\n- Sliding-Window Attention\n- Byte-fallback BPE tokenizer\n\n## Troubleshooting\n- If you see the following error:\n```\nTraceback (most recent call last):\nFile \"\", line 1, in\nFile \"/transformers/models/auto/auto_factory.py\", line 482, in from_pretrained\nconfig, kwargs = AutoConfig.from_pretrained(\nFile \"/transformers/models/auto/configuration_auto.py\", line 1022, in from_pretrained\nconfig_class = CONFIG_MAPPING[config_dict[\"model_type\"]]\nFile \"/transformers/models/auto/configuration_auto.py\", line 723, in getitem\nraise KeyError(key)\nKeyError: 'mistral'\n```\n\nInstalling transformers from source should solve the issue\npip install git+https://github.com/huggingface/transformers\n\nThis should not be required after transformers-v4.33.4.\n\n## Limitations\n\nThe Mistral 7B Instruct model is a quick demonstration that the base model can be easily fine-tuned to achieve compelling performance. \nIt does not have any moderation mechanisms. We're looking forward to engaging with the community on ways to\nmake the model finely respect guardrails, allowing for deployment in environments requiring moderated outputs.\n\n## The Mistral AI Team\n\nAlbert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.",
        "llm_extraction": {
            "model_name": "Mistral-7B-Instruct-v0.1",
            "model_framework": "transformers",
            "model_architecture": "Mistral-7B-v0.1",
            "tasks": [
                "text-generation"
            ],
            "training_strategy": "instruct fine-tuning",
            "parameters": "temperature: 0.7",
            "vocab_size": "50,257",
            "training_data": "",
            "authors": [
                "Albert Jiang",
                "Alexandre Sablayrolles",
                "Arthur Mensch",
                "Chris Bamford",
                "Devendra Singh Chaplot",
                "Diego de las Casas",
                "Florian Bressand",
                "Gianna Lengyel",
                "Guillaume Lample",
                "Lélio Renard Lavaud",
                "Lucile Saulnier",
                "Marie-Anne Lachaux",
                "Pierre Stock",
                "Teven Le Scao",
                "Thibaut Lavril",
                "Thomas Wang",
                "Timothée Lacroix",
                "William El Sayed"
            ],
            "other": [
                "Byte-fallback BPE tokenizer"
            ]
        },
        "truncation": 0,
        "extraction_version": "v_1",
        "papers_with_code": "Performance not found on papers with code",
        "model_usage": {
            "llm_input": {
                "Model Card for Mistral-7B-Instruct-v0.1": {
                    "Model Card for Mistral-7B-Instruct-v0.1/ Instruction format": "In order to leverage instruction fine-tuning, your prompt should be surrounded by [INST] and [/INST] tokens. The very first instruction should begin with a begin of sentence id. The next instructions should not. The assistant generation will be ended by the end-of-sentence token id.\nE.g.\ntext = \"<s>[INST] What is your favourite condiment? [/INST]\"\n\"Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> \"\n\"[INST] Do you have mayonnaise recipes? [/INST]\"\nThis format is available as a chat template via the apply_chat_template() method:",
                    "Model Card for Mistral-7B-Instruct-v0.1/ Model Architecture": "This instruction model is based on Mistral-7B-v0.1, a transformer model with the following architecture choices:\n['Grouped-Query Attention', 'Sliding-Window Attention', 'Byte-fallback BPE tokenizer']",
                    "Model Card for Mistral-7B-Instruct-v0.1/ Troubleshooting": [
                        "If you see the following error:"
                    ],
                    "Model Card for Mistral-7B-Instruct-v0.1/ Limitations": "The Mistral 7B Instruct model is a quick demonstration that the base model can be easily fine-tuned to achieve compelling performance. \nIt does not have any moderation mechanisms. We're looking forward to engaging with the community on ways to\nmake the model finely respect guardrails, allowing for deployment in environments requiring moderated outputs.",
                    "Model Card for Mistral-7B-Instruct-v0.1/ The Mistral AI Team": "Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed."
                }
            },
            "usage": {},
            "model_function": []
        }
    },
    "mistralai/Mistral-7B-v0.1": {
        "model_name": "Mistral-7B-v0.1",
        "org": "mistralai",
        "model_info": {
            "id": "mistralai/Mistral-7B-v0.1",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 944080,
            "likes": 2771,
            "library_name": "transformers",
            "tags": [
                "transformers",
                "pytorch",
                "safetensors",
                "mistral",
                "text-generation",
                "pretrained",
                "en",
                "arxiv:2310.06825",
                "license:apache-2.0",
                "autotrain_compatible",
                "endpoints_compatible",
                "has_space",
                "text-generation-inference",
                "region:us"
            ],
            "pipeline_tag": "text-generation",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "650aedb6238a644cb93a52c3",
            "createdAt": "2023-09-20T13:03:50.000Z",
            "modelId": "mistralai/Mistral-7B-v0.1"
        },
        "card_to_dict": {
            "language": [
                "en"
            ],
            "license": "apache-2.0",
            "tags": [
                "pretrained"
            ],
            "pipeline_tag": "text-generation",
            "inference": {
                "parameters": {
                    "temperature": 0.7
                }
            }
        },
        "relevant_websites": [
            "https://arxiv.org/abs/2310.06825",
            "https://mistral.ai/news/announcing-mistral-7b"
        ],
        "text": "language: - en license: apache-2.0 tags: - pretrained pipeline_tag: text-generation inference:   parameters:     temperature: 0.7  Model Card for Mistral-7B-v0.1 The Mistral-7B-v0.1 Large Language Model (LLM) is a pretrained generative text model with 7 billion parameters.  Mistral-7B-v0.1 outperforms Llama 2 13B on all benchmarks we tested. For full details of this model please read our paper and release blog post. Model Architecture Mistral-7B-v0.1 is a transformer model, with the following architecture choices: - Grouped-Query Attention - Sliding-Window Attention - Byte-fallback BPE tokenizer Troubleshooting  If you see the following error: KeyError: 'mistral' Or: NotImplementedError: Cannot copy out of meta tensor; no data!  Ensure you are utilizing a stable version of Transformers, 4.34.0 or newer. Notice Mistral 7B is a pretrained base model and therefore does not have any moderation mechanisms. The Mistral AI Team Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.",
        "markdown_text": "---\nlanguage:\n- en\nlicense: apache-2.0\ntags:\n- pretrained\npipeline_tag: text-generation\ninference:\n  parameters:\n    temperature: 0.7\n---\n\n# Model Card for Mistral-7B-v0.1\n\nThe Mistral-7B-v0.1 Large Language Model (LLM) is a pretrained generative text model with 7 billion parameters. \nMistral-7B-v0.1 outperforms Llama 2 13B on all benchmarks we tested.\n\nFor full details of this model please read our [paper](https://arxiv.org/abs/2310.06825) and [release blog post](https://mistral.ai/news/announcing-mistral-7b/).\n\n## Model Architecture\n\nMistral-7B-v0.1 is a transformer model, with the following architecture choices:\n- Grouped-Query Attention\n- Sliding-Window Attention\n- Byte-fallback BPE tokenizer\n\n## Troubleshooting\n\n- If you see the following error:\n```\nKeyError: 'mistral'\n```\n- Or:\n```\nNotImplementedError: Cannot copy out of meta tensor; no data!\n```\n\nEnsure you are utilizing a stable version of Transformers, 4.34.0 or newer.\n\n## Notice\n\nMistral 7B is a pretrained base model and therefore does not have any moderation mechanisms.\n\n## The Mistral AI Team\n \nAlbert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed.",
        "llm_extraction": {
            "model_name": "Mistral-7B-v0.1",
            "model_framework": "transformers",
            "model_architecture": "transformer",
            "tasks": [
                "text-generation"
            ],
            "training_strategy": "inference",
            "parameters": {
                "temperature": 0.7
            },
            "vocab_size": "NONE",
            "training_data": "NONE",
            "authors": [
                "Albert Jiang",
                "Alexandre Sablayrolles",
                "Arthur Mensch",
                "Chris Bamford",
                "Devendra Singh Chaplot",
                "Diego de las Casas",
                "Florian Bressand",
                "Gianna Lengyel",
                "Guillaume Lample",
                "Lélio Renard Lavaud",
                "Lucile Saulnier",
                "Marie-Anne Lachaux",
                "Pierre Stock",
                "Teven Le Scao",
                "Thibaut Lavril",
                "Thomas Wang",
                "Timothée Lacroix",
                "William El Sayed"
            ],
            "other": [
                "Byte-fallback BPE tokenizer"
            ]
        },
        "truncation": 0,
        "extraction_version": "v_1",
        "papers_with_code": "successful",
        "performance": [
            {
                "id": "f625b333-c3e3-415a-843c-a370dbb0ee0d",
                "best_rank": NaN,
                "metrics": {
                    "Accuracy": "55.5"
                },
                "methodology": "Mistral 7B (0-shot)",
                "uses_additional_data": false,
                "paper": "mistral-7b",
                "best_metric": null,
                "evaluated_on": "2023-10-10",
                "evaluation": "common-sense-reasoning-on-arc-challenge",
                "benchmark_details": {
                    "id": "common-sense-reasoning-on-arc-challenge",
                    "task": "common-sense-reasoning",
                    "dataset": "arc-challenge",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "d7e71a61-fd23-4dd4-b9db-239be384f935",
                "best_rank": NaN,
                "metrics": {
                    "Accuracy": "80.0"
                },
                "methodology": "Mistral 7B (0-shot)",
                "uses_additional_data": false,
                "paper": "mistral-7b",
                "best_metric": null,
                "evaluated_on": "2023-10-10",
                "evaluation": "common-sense-reasoning-on-arc-easy",
                "benchmark_details": {
                    "id": "common-sense-reasoning-on-arc-easy",
                    "task": "common-sense-reasoning",
                    "dataset": "arc-easy",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "da4cfcdc-0c9d-4ae0-85c2-ac04efea2a39",
                "best_rank": NaN,
                "metrics": {
                    "Accuracy": "52.2",
                    "Parameters (Billion)": "7"
                },
                "methodology": "Mistral 7B (maj@8)",
                "uses_additional_data": false,
                "paper": "mistral-7b",
                "best_metric": null,
                "evaluated_on": "2023-10-10",
                "evaluation": "arithmetic-reasoning-on-gsm8k",
                "benchmark_details": {
                    "id": "arithmetic-reasoning-on-gsm8k",
                    "task": "arithmetic-reasoning",
                    "dataset": "gsm8k",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "05f8c7b2-8f8b-4e02-ab60-e1076bc13dd8",
                "best_rank": NaN,
                "metrics": {
                    "Accuracy": "81.3"
                },
                "methodology": "Mistral 7B (0-shot)",
                "uses_additional_data": false,
                "paper": "mistral-7b",
                "best_metric": null,
                "evaluated_on": "2023-10-10",
                "evaluation": "sentence-completion-on-hellaswag",
                "benchmark_details": {
                    "id": "sentence-completion-on-hellaswag",
                    "task": "sentence-completion",
                    "dataset": "hellaswag",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "6c354850-2108-4562-b408-dd048c0c29f6",
                "best_rank": NaN,
                "metrics": {
                    "Pass@1": "30.5"
                },
                "methodology": "Mistral 7B (0-shot)",
                "uses_additional_data": true,
                "paper": "mistral-7b",
                "best_metric": null,
                "evaluated_on": "2023-10-10",
                "evaluation": "code-generation-on-humaneval",
                "benchmark_details": {
                    "id": "code-generation-on-humaneval",
                    "task": "code-generation",
                    "dataset": "humaneval",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "884a02a3-223a-4ba9-b79a-5f249f3bf0ae",
                "best_rank": NaN,
                "metrics": {
                    "Accuracy": "50.4"
                },
                "methodology": "Mistral (7B)",
                "uses_additional_data": false,
                "paper": "mistral-7b",
                "best_metric": null,
                "evaluated_on": "2023-10-10",
                "evaluation": "zero-shot-video-question-answer-on-intentqa",
                "benchmark_details": {
                    "id": "zero-shot-video-question-answer-on-intentqa",
                    "task": "zeroshot-video-question-answer",
                    "dataset": "intentqa",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "815433a3-d31f-4294-8c87-435b8d45d4b0",
                "best_rank": NaN,
                "metrics": {
                    "Accuracy": "13.1",
                    "Parameters (Billions)": "7"
                },
                "methodology": "Mistral 7B (maj@4)",
                "uses_additional_data": false,
                "paper": "mistral-7b",
                "best_metric": null,
                "evaluated_on": "2023-10-10",
                "evaluation": "math-word-problem-solving-on-math",
                "benchmark_details": {
                    "id": "math-word-problem-solving-on-math",
                    "task": "math-word-problem-solving",
                    "dataset": "math",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "64cbd1d1-9f54-450d-89a9-807c44e429fe",
                "best_rank": NaN,
                "metrics": {
                    "Accuracy": "47.5"
                },
                "methodology": "Mistral 7B (3-shot)",
                "uses_additional_data": false,
                "paper": "mistral-7b",
                "best_metric": null,
                "evaluated_on": "2023-10-10",
                "evaluation": "code-generation-on-mbpp",
                "benchmark_details": {
                    "id": "code-generation-on-mbpp",
                    "task": "code-generation",
                    "dataset": "mbpp",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "55fb770f-e155-4967-8f36-a2cccc773b12",
                "best_rank": NaN,
                "metrics": {
                    "Average (%)": "60.1"
                },
                "methodology": "Mistral 7B (5-shot)",
                "uses_additional_data": false,
                "paper": "mistral-7b",
                "best_metric": null,
                "evaluated_on": "2023-10-10",
                "evaluation": "multi-task-language-understanding-on-mmlu",
                "benchmark_details": {
                    "id": "multi-task-language-understanding-on-mmlu",
                    "task": "multi-task-language-understanding",
                    "dataset": "mmlu",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "1e27e11a-3cc5-4d50-be3f-7d4c699bf0c2",
                "best_rank": NaN,
                "metrics": {
                    "EM": "28.8"
                },
                "methodology": "Mistral 7B (5-shot)",
                "uses_additional_data": false,
                "paper": "mistral-7b",
                "best_metric": null,
                "evaluated_on": "2023-10-10",
                "evaluation": "question-answering-on-natural-questions",
                "benchmark_details": {
                    "id": "question-answering-on-natural-questions",
                    "task": "question-answering",
                    "dataset": "natural-questions",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "6f9a52b1-4cb4-4f31-b010-56a6298ea585",
                "best_rank": 4.0,
                "metrics": {
                    "Acc@GQA": "9.2"
                },
                "methodology": "Mistral (7B)",
                "uses_additional_data": false,
                "paper": "mistral-7b",
                "best_metric": "Acc@GQA",
                "evaluated_on": "2023-10-10",
                "evaluation": "zero-shot-video-question-answer-on-next-gqa",
                "benchmark_details": {
                    "id": "zero-shot-video-question-answer-on-next-gqa",
                    "task": "zeroshot-video-question-answer",
                    "dataset": "next-gqa",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "d0ebfb44-6709-4405-b3cc-11f148f445da",
                "best_rank": NaN,
                "metrics": {
                    "Accuracy": "51.1"
                },
                "methodology": "Mistral (7B)",
                "uses_additional_data": false,
                "paper": "mistral-7b",
                "best_metric": null,
                "evaluated_on": "2023-10-10",
                "evaluation": "zero-shot-video-question-answer-on-next-qa",
                "benchmark_details": {
                    "id": "zero-shot-video-question-answer-on-next-qa",
                    "task": "zeroshot-video-question-answer",
                    "dataset": "next-qa",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "65369886-8bc1-4447-b82b-40f443e4f012",
                "best_rank": 11.0,
                "metrics": {
                    "Accuracy": "83.0"
                },
                "methodology": "Mistral 7B (0-shot)",
                "uses_additional_data": false,
                "paper": "mistral-7b",
                "best_metric": "Accuracy",
                "evaluated_on": "2023-10-10",
                "evaluation": "question-answering-on-piqa",
                "benchmark_details": {
                    "id": "question-answering-on-piqa",
                    "task": "question-answering",
                    "dataset": "piqa",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "279619c2-43d9-4dd4-a19a-14ab94e3be0b",
                "best_rank": NaN,
                "metrics": {
                    "EM": "69.9"
                },
                "methodology": "Mistral 7B (5-shot)",
                "uses_additional_data": false,
                "paper": "mistral-7b",
                "best_metric": null,
                "evaluated_on": "2023-10-10",
                "evaluation": "question-answering-on-triviaqa",
                "benchmark_details": {
                    "id": "question-answering-on-triviaqa",
                    "task": "question-answering",
                    "dataset": "triviaqa",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "f88bb943-c1b8-4ee1-a242-a6794468b72d",
                "best_rank": NaN,
                "metrics": {
                    "Accuracy": "75.3"
                },
                "methodology": "Mistral 7B (0-shot)",
                "uses_additional_data": false,
                "paper": "mistral-7b",
                "best_metric": null,
                "evaluated_on": "2023-10-10",
                "evaluation": "common-sense-reasoning-on-winogrande",
                "benchmark_details": {
                    "id": "common-sense-reasoning-on-winogrande",
                    "task": "common-sense-reasoning",
                    "dataset": "winogrande",
                    "description": "",
                    "mirror_url": null
                }
            }
        ],
        "model_usage": {
            "llm_input": {
                "Model Card for Mistral-7B-v0.1": {
                    "Model Card for Mistral-7B-v0.1/ Model Architecture": "Mistral-7B-v0.1 is a transformer model, with the following architecture choices:\n['Grouped-Query Attention', 'Sliding-Window Attention', 'Byte-fallback BPE tokenizer']",
                    "Model Card for Mistral-7B-v0.1/ Troubleshooting": [
                        "If you see the following error:"
                    ],
                    "Model Card for Mistral-7B-v0.1/ Notice": "Mistral 7B is a pretrained base model and therefore does not have any moderation mechanisms.",
                    "Model Card for Mistral-7B-v0.1/ The Mistral AI Team": "Albert Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed."
                }
            },
            "usage": {},
            "model_function": []
        }
    },
    "nvidia/segformer-b0-finetuned-ade-512-512": {
        "model_name": "segformer-b0-finetuned-ade-512-512",
        "org": "nvidia",
        "model_info": {
            "id": "nvidia/segformer-b0-finetuned-ade-512-512",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 53967,
            "likes": 101,
            "library_name": "transformers",
            "tags": [
                "transformers",
                "pytorch",
                "tf",
                "safetensors",
                "segformer",
                "vision",
                "image-segmentation",
                "dataset:scene_parse_150",
                "arxiv:2105.15203",
                "license:other",
                "endpoints_compatible",
                "has_space",
                "region:us"
            ],
            "pipeline_tag": "image-segmentation",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "621ffdc136468d709f17e966",
            "createdAt": "2022-03-02T23:29:05.000Z",
            "modelId": "nvidia/segformer-b0-finetuned-ade-512-512"
        },
        "card_to_dict": {
            "license": "other",
            "tags": [
                "vision",
                "image-segmentation"
            ],
            "datasets": [
                "scene_parse_150"
            ],
            "widget": [
                {
                    "src": "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg",
                    "example_title": "House"
                },
                {
                    "src": "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000002.jpg",
                    "example_title": "Castle"
                }
            ]
        },
        "relevant_websites": [
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg",
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000002.jpg",
            "https://arxiv.org/abs/2105.15203",
            "https://github.com/NVlabs/SegFormer",
            "https://huggingface.co/models?other=segformer",
            "https://huggingface.co/transformers/model_doc/segformer.html",
            "https://github.com/NVlabs/SegFormer/blob/master/LICENSE",
            "https://arxiv.org/abs/2105.15203",
            "https://dblp.org/rec/journals/corr/abs-2105-15203.bib",
            "https://dblp.org"
        ],
        "text": "license: other tags: - vision - image-segmentation datasets: - scene_parse_150 widget: - src:    example_title: House - src:    example_title: Castle  SegFormer (b0-sized) model fine-tuned on ADE20k SegFormer model fine-tuned on ADE20k at resolution 512x512. It was introduced in the paper SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers by Xie et al. and first released in this repository.  Disclaimer: The team releasing SegFormer did not write a model card for this model so this model card has been written by the Hugging Face team. Model description SegFormer consists of a hierarchical Transformer encoder and a lightweight all-MLP decode head to achieve great results on semantic segmentation benchmarks such as ADE20K and Cityscapes. The hierarchical Transformer is first pre-trained on ImageNet-1k, after which a decode head is added and fine-tuned altogether on a downstream dataset. Intended uses & limitations You can use the raw model for semantic segmentation. See the model hub to look for fine-tuned versions on a task that interests you. How to use Here is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes: For more code examples, we refer to the documentation. License The license for this model can be found here. BibTeX entry and citation info bibtex @article{DBLP:journals/corr/abs-2105-15203,   author    = {Enze Xie and                Wenhai Wang and                Zhiding Yu and                Anima Anandkumar and                Jose M. Alvarez and                Ping Luo},   title     = {SegFormer: Simple and Efficient Design for Semantic Segmentation with                Transformers},   journal   = {CoRR},   volume    = {abs/2105.15203},   year      = {2021},   url       = {},   eprinttype = {arXiv},   eprint    = {2105.15203},   timestamp = {Wed, 02 Jun 2021 11:46:42 +0200},   biburl    = {},   bibsource = {dblp computer science bibliography, } }",
        "markdown_text": "---\nlicense: other\ntags:\n- vision\n- image-segmentation\ndatasets:\n- scene_parse_150\nwidget:\n- src: https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg\n  example_title: House\n- src: https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000002.jpg\n  example_title: Castle\n---\n\n# SegFormer (b0-sized) model fine-tuned on ADE20k\n\nSegFormer model fine-tuned on ADE20k at resolution 512x512. It was introduced in the paper [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203) by Xie et al. and first released in [this repository](https://github.com/NVlabs/SegFormer). \n\nDisclaimer: The team releasing SegFormer did not write a model card for this model so this model card has been written by the Hugging Face team.\n\n## Model description\n\nSegFormer consists of a hierarchical Transformer encoder and a lightweight all-MLP decode head to achieve great results on semantic segmentation benchmarks such as ADE20K and Cityscapes. The hierarchical Transformer is first pre-trained on ImageNet-1k, after which a decode head is added and fine-tuned altogether on a downstream dataset.\n\n## Intended uses & limitations\n\nYou can use the raw model for semantic segmentation. See the [model hub](https://huggingface.co/models?other=segformer) to look for fine-tuned versions on a task that interests you.\n\n### How to use\n\nHere is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes:\n\n```python\nfrom transformers import SegformerImageProcessor, SegformerForSemanticSegmentation\nfrom PIL import Image\nimport requests\n\nprocessor = SegformerImageProcessor.from_pretrained(\"nvidia/segformer-b0-finetuned-ade-512-512\")\nmodel = SegformerForSemanticSegmentation.from_pretrained(\"nvidia/segformer-b0-finetuned-ade-512-512\")\n\nurl = \"http://images.cocodataset.org/val2017/000000039769.jpg\"\nimage = Image.open(requests.get(url, stream=True).raw)\n\ninputs = feature_extractor(images=image, return_tensors=\"pt\")\noutputs = model(**inputs)\nlogits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)\n```\n\nFor more code examples, we refer to the [documentation](https://huggingface.co/transformers/model_doc/segformer.html#).\n\n### License\n\nThe license for this model can be found [here](https://github.com/NVlabs/SegFormer/blob/master/LICENSE).\n\n### BibTeX entry and citation info\n\n```bibtex\n@article{DBLP:journals/corr/abs-2105-15203,\n  author    = {Enze Xie and\n               Wenhai Wang and\n               Zhiding Yu and\n               Anima Anandkumar and\n               Jose M. Alvarez and\n               Ping Luo},\n  title     = {SegFormer: Simple and Efficient Design for Semantic Segmentation with\n               Transformers},\n  journal   = {CoRR},\n  volume    = {abs/2105.15203},\n  year      = {2021},\n  url       = {https://arxiv.org/abs/2105.15203},\n  eprinttype = {arXiv},\n  eprint    = {2105.15203},\n  timestamp = {Wed, 02 Jun 2021 11:46:42 +0200},\n  biburl    = {https://dblp.org/rec/journals/corr/abs-2105-15203.bib},\n  bibsource = {dblp computer science bibliography, https://dblp.org}\n}\n```\n",
        "llm_extraction": {
            "model_name": "SegFormer",
            "model_framework": "transformers",
            "model_architecture": "hierarchical Transformer encoder and a lightweight all-MLP decode head",
            "tasks": [
                "semantic segmentation"
            ],
            "training_strategy": "fine-tuned on ADE20k",
            "parameters": "b0-sized",
            "vocab_size": "NONE",
            "training_data": "ADE20k",
            "authors": [
                "Xie et al."
            ],
            "other": [
                "ImageNet-1k",
                "ADE20K",
                "Cityscapes",
                "raw model",
                "fine-tuned versions",
                "citation info"
            ]
        },
        "truncation": 0,
        "extraction_version": "v_1",
        "papers_with_code": "successful",
        "performance": [
            {
                "id": "d091dcb8-2311-42fe-b31c-0258b0491a9b",
                "best_rank": NaN,
                "metrics": {
                    "mIoU": "57.20"
                },
                "methodology": "SegFormer",
                "uses_additional_data": false,
                "paper": "segformer-simple-and-efficient-design-for",
                "best_metric": null,
                "evaluated_on": "2021-05-31",
                "evaluation": "semantic-segmentation-on-deliver-1",
                "benchmark_details": {
                    "id": "semantic-segmentation-on-deliver-1",
                    "task": "semantic-segmentation",
                    "dataset": "deliver",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "5e33322b-3756-433b-a97a-ef69e1306ef6",
                "best_rank": NaN,
                "metrics": {
                    "MAE": "0.053"
                },
                "methodology": "SegFormer",
                "uses_additional_data": false,
                "paper": "segformer-simple-and-efficient-design-for",
                "best_metric": null,
                "evaluated_on": "2021-05-31",
                "evaluation": "thermal-image-segmentation-on-rgb-t-glass",
                "benchmark_details": {
                    "id": "thermal-image-segmentation-on-rgb-t-glass",
                    "task": "thermal-image-segmentation",
                    "dataset": "rgb-t-glass-segmentation",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "5916e77b-6efa-4e2f-bd06-5f8669dbfc66",
                "best_rank": 2.0,
                "metrics": {
                    "mIoU": "77.2"
                },
                "methodology": "SegFormer",
                "uses_additional_data": false,
                "paper": "segformer-simple-and-efficient-design-for",
                "best_metric": "mIoU",
                "evaluated_on": "2021-05-31",
                "evaluation": "semantic-segmentation-on-selma",
                "benchmark_details": {
                    "id": "semantic-segmentation-on-selma",
                    "task": "semantic-segmentation",
                    "dataset": "selma",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "4f420f06-3f85-44e8-ac5a-19639de73885",
                "best_rank": NaN,
                "metrics": {
                    "mIoU (Real)": "82.20",
                    "mIoU (Syn)": "78.53"
                },
                "methodology": "SegFormer",
                "uses_additional_data": false,
                "paper": "segformer-simple-and-efficient-design-for",
                "best_metric": null,
                "evaluated_on": "2021-05-31",
                "evaluation": "semantic-segmentation-on-urbanlf",
                "benchmark_details": {
                    "id": "semantic-segmentation-on-urbanlf",
                    "task": "semantic-segmentation",
                    "dataset": "urbanlf",
                    "description": "",
                    "mirror_url": null
                }
            }
        ],
        "model_usage": {
            "llm_input": {
                "SegFormer (b0-sized) model fine-tuned on ADE20k": {
                    "SegFormer (b0-sized) model fine-tuned on ADE20k/ Model description": "SegFormer consists of a hierarchical Transformer encoder and a lightweight all-MLP decode head to achieve great results on semantic segmentation benchmarks such as ADE20K and Cityscapes. The hierarchical Transformer is first pre-trained on ImageNet-1k, after which a decode head is added and fine-tuned altogether on a downstream dataset.",
                    "Intended uses & limitations/ License": "The license for this model can be found here.",
                    "Intended uses & limitations/ BibTeX entry and citation info": "@article{DBLP:journals/corr/abs-2105-15203,\n  author    = {Enze Xie and\n               Wenhai Wang and\n               Zhiding Yu and\n               Anima Anandkumar and\n               Jose M. Alvarez and\n               Ping Luo},\n  title     = {SegFormer: Simple and Efficient Design for Semantic Segmentation with\n               Transformers},\n  journal   = {CoRR},\n  volume    = {abs/2105.15203},\n  year      = {2021},\n  url       = {https://arxiv.org/abs/2105.15203},\n  eprinttype = {arXiv},\n  eprint    = {2105.15203},\n  timestamp = {Wed, 02 Jun 2021 11:46:42 +0200},\n  biburl    = {https://dblp.org/rec/journals/corr/abs-2105-15203.bib},\n  bibsource = {dblp computer science bibliography, https://dblp.org}\n}"
                }
            },
            "usage": {
                "Intended uses & limitations/ How to use": "Here is how to use this model to classify an image of the COCO 2017 dataset into one of the 1,000 ImageNet classes:\n```\nfrom transformers import SegformerImageProcessor, SegformerForSemanticSegmentation\nfrom PIL import Image\nimport requests\nprocessor = SegformerImageProcessor.from_pretrained(\"nvidia/segformer-b0-finetuned-ade-512-512\")\nmodel = SegformerForSemanticSegmentation.from_pretrained(\"nvidia/segformer-b0-finetuned-ade-512-512\")\nurl = \"http://images.cocodataset.org/val2017/000000039769.jpg\"\nimage = Image.open(requests.get(url, stream=True).raw)\ninputs = feature_extractor(images=image, return_tensors=\"pt\")\noutputs = model(**inputs)\nlogits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)\n```\nFor more code examples, we refer to the documentation."
            },
            "model_function": [
                {
                    "code": "from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation\nfrom PIL import Image\nimport requests\n\ndef classify_image(image_url, model_name=\"nvidia/segformer-b0-finetuned-ade-512-512\"):\n    processor = SegformerImageProcessor.from_pretrained(model_name)\n    model = SegformerForSemanticSegmentation.from_pretrained(model_name)\n    image = Image.open(requests.get(image_url, stream=True).raw)\n    inputs = processor(images=image, return_tensors=\"pt\")\n    outputs = model(**inputs)\n    logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)\n    return logits",
                    "function_info": {
                        "return": null,
                        "function_name": "classify_image"
                    }
                }
            ]
        }
    },
    "openai/whisper-large-v2": {
        "model_name": "whisper-large-v2",
        "org": "openai",
        "model_info": {
            "id": "openai/whisper-large-v2",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 464150,
            "likes": 1503,
            "library_name": "transformers",
            "tags": [
                "transformers",
                "pytorch",
                "tf",
                "jax",
                "safetensors",
                "whisper",
                "automatic-speech-recognition",
                "audio",
                "hf-asr-leaderboard",
                "en",
                "zh",
                "de",
                "es",
                "ru",
                "ko",
                "fr",
                "ja",
                "pt",
                "tr",
                "pl",
                "ca",
                "nl",
                "ar",
                "sv",
                "it",
                "id",
                "hi",
                "fi",
                "vi",
                "he",
                "uk",
                "el",
                "ms",
                "cs",
                "ro",
                "da",
                "hu",
                "ta",
                "no",
                "th",
                "ur",
                "hr",
                "bg",
                "lt",
                "la",
                "mi",
                "ml",
                "cy",
                "sk",
                "te",
                "fa",
                "lv",
                "bn",
                "sr",
                "az",
                "sl",
                "kn",
                "et",
                "mk",
                "br",
                "eu",
                "is",
                "hy",
                "ne",
                "mn",
                "bs",
                "kk",
                "sq",
                "sw",
                "gl",
                "mr",
                "pa",
                "si",
                "km",
                "sn",
                "yo",
                "so",
                "af",
                "oc",
                "ka",
                "be",
                "tg",
                "sd",
                "gu",
                "am",
                "yi",
                "lo",
                "uz",
                "fo",
                "ht",
                "ps",
                "tk",
                "nn",
                "mt",
                "sa",
                "lb",
                "my",
                "bo",
                "tl",
                "mg",
                "as",
                "tt",
                "haw",
                "ln",
                "ha",
                "ba",
                "jw",
                "su",
                "arxiv:2212.04356",
                "license:apache-2.0",
                "endpoints_compatible",
                "has_space",
                "region:us"
            ],
            "pipeline_tag": "automatic-speech-recognition",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "638e3b8c629b4d0a62cd6dcc",
            "createdAt": "2022-12-05T18:42:20.000Z",
            "modelId": "openai/whisper-large-v2"
        },
        "card_to_dict": {
            "language": [
                "en",
                "zh",
                "de",
                "es",
                "ru",
                "ko",
                "fr",
                "ja",
                "pt",
                "tr",
                "pl",
                "ca",
                "nl",
                "ar",
                "sv",
                "it",
                "id",
                "hi",
                "fi",
                "vi",
                "he",
                "uk",
                "el",
                "ms",
                "cs",
                "ro",
                "da",
                "hu",
                "ta",
                false,
                "th",
                "ur",
                "hr",
                "bg",
                "lt",
                "la",
                "mi",
                "ml",
                "cy",
                "sk",
                "te",
                "fa",
                "lv",
                "bn",
                "sr",
                "az",
                "sl",
                "kn",
                "et",
                "mk",
                "br",
                "eu",
                "is",
                "hy",
                "ne",
                "mn",
                "bs",
                "kk",
                "sq",
                "sw",
                "gl",
                "mr",
                "pa",
                "si",
                "km",
                "sn",
                "yo",
                "so",
                "af",
                "oc",
                "ka",
                "be",
                "tg",
                "sd",
                "gu",
                "am",
                "yi",
                "lo",
                "uz",
                "fo",
                "ht",
                "ps",
                "tk",
                "nn",
                "mt",
                "sa",
                "lb",
                "my",
                "bo",
                "tl",
                "mg",
                "as",
                "tt",
                "haw",
                "ln",
                "ha",
                "ba",
                "jw",
                "su"
            ],
            "license": "apache-2.0",
            "tags": [
                "audio",
                "automatic-speech-recognition",
                "hf-asr-leaderboard"
            ],
            "widget": [
                {
                    "example_title": "Librispeech sample 1",
                    "src": "https://cdn-media.huggingface.co/speech_samples/sample1.flac"
                },
                {
                    "example_title": "Librispeech sample 2",
                    "src": "https://cdn-media.huggingface.co/speech_samples/sample2.flac"
                }
            ],
            "pipeline_tag": "automatic-speech-recognition"
        },
        "relevant_websites": [
            "https://cdn-media.huggingface.co/speech_samples/sample1.flac",
            "https://cdn-media.huggingface.co/speech_samples/sample2.flac",
            "https://arxiv.org/abs/2212.04356",
            "https://github.com/openai/whisper",
            "https://huggingface.co/models?search=openai/whisper",
            "https://huggingface.co/openai/whisper-tiny.en",
            "https://huggingface.co/openai/whisper-tiny",
            "https://huggingface.co/openai/whisper-base.en",
            "https://huggingface.co/openai/whisper-base",
            "https://huggingface.co/openai/whisper-small.en",
            "https://huggingface.co/openai/whisper-small",
            "https://huggingface.co/openai/whisper-medium.en",
            "https://huggingface.co/openai/whisper-medium",
            "https://huggingface.co/openai/whisper-large",
            "https://huggingface.co/openai/whisper-large-v2",
            "https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperProcessor",
            "https://huggingface.co/datasets/librispeech_asr",
            "https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline",
            "https://huggingface.co/blog/asr-chunking",
            "https://huggingface.co/blog/fine-tune-whisper",
            "https://cdn.openai.com/papers/whisper.pdf",
            "https://cdn.openai.com/papers/whisper.pdf",
            "https://cdn.openai.com/papers/whisper.pdf",
            "https://arxiv.org/abs/2212.04356"
        ],
        "text": "language: - en - zh - de - es - ru - ko - fr - ja - pt - tr - pl - ca - nl - ar - sv - it - id - hi - fi - vi - he - uk - el - ms - cs - ro - da - hu - ta - false - th - ur - hr - bg - lt - la - mi - ml - cy - sk - te - fa - lv - bn - sr - az - sl - kn - et - mk - br - eu - is - hy - ne - mn - bs - kk - sq - sw - gl - mr - pa - si - km - sn - yo - so - af - oc - ka - be - tg - sd - gu - am - yi - lo - uz - fo - ht - ps - tk - nn - mt - sa - lb - my - bo - tl - mg - as - tt - haw - ln - ha - ba - jw - su license: apache-2.0 tags: - audio - automatic-speech-recognition - hf-asr-leaderboard widget: - example_title: Librispeech sample 1   src:  - example_title: Librispeech sample 2   src:  pipeline_tag: automatic-speech-recognition  Whisper Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. Trained on 680k hours  of labelled data, Whisper models demonstrate a strong ability to generalise to many datasets and domains without the need  for fine-tuning. Whisper was proposed in the paper Robust Speech Recognition via Large-Scale Weak Supervision  by Alec Radford et al. from OpenAI. The original code repository can be found here. Compared to the Whisper large model, the large-v2 model is trained for 2.5x more epochs with added regularization  for improved performance. Disclaimer: Content for this model card has partly been written by the Hugging Face team, and parts of it were  copied and pasted from the original model card. Model details Whisper is a Transformer based encoder-decoder model, also referred to as a sequence-to-sequence model.  It was trained on 680k hours of labelled speech data annotated using large-scale weak supervision.  The models were trained on either English-only data or multilingual data. The English-only models were trained  on the task of speech recognition. The multilingual models were trained on both speech recognition and speech  translation. For speech recognition, the model predicts transcriptions in the same language as the audio.  For speech translation, the model predicts transcriptions to a different language to the audio. Whisper checkpoints come in five configurations of varying model sizes. The smallest four are trained on either English-only or multilingual data. The largest checkpoints are multilingual only. All ten of the pre-trained checkpoints  are available on the Hugging Face Hub. The  checkpoints are summarised in the following table with links to the models on the Hub: Usage To transcribe audio samples, the model has to be used alongside a WhisperProcessor. The WhisperProcessor is used to: 1. Pre-process the audio inputs (converting them to log-Mel spectrograms for the model) 2. Post-process the model outputs (converting them from tokens to text) The model is informed of which task to perform (transcription or translation) by passing the appropriate \"context tokens\". These context tokens  are a sequence of tokens that are given to the decoder at the start of the decoding process, and take the following order: 1. The transcription always starts with the <> token 2. The second token is the language token (e.g. <> for English) 3. The third token is the \"task token\". It can take one of two values: <> for speech translation 4. In addition, a <> token is added if the model should not include timestamp prediction Thus, a typical sequence of context tokens might look as follows: <> Which tells the model to decode in English, under the task of speech recognition, and not to predict timestamps. These tokens can either be forced or un-forced. If they are forced, the model is made to predict each token at  each position. This allows one to control the output language and task for the Whisper model. If they are un-forced,  the Whisper model will automatically predict the output langauge and task itself. The context tokens can be set accordingly: Which forces the model to predict in English under the task of speech recognition. Transcription English to English In this example, the context tokens are 'unforced', meaning the model automatically predicts the output language (English) and task (transcribe). The context tokens can be removed from the start of the transcription by setting skip_special_tokens=True. French to French The following example demonstrates French to French transcription by setting the decoder ids appropriately.  Translation Setting the task to \"translate\" forces the Whisper model to perform speech translation. French to English Evaluation This code snippet shows how to evaluate Whisper Large on LibriSpeech test-clean: Long-Form Transcription The Whisper model is intrinsically designed to work on audio samples of up to 30s in duration. However, by using a chunking  algorithm, it can be used to transcribe audio samples of up to arbitrary length. This is possible through Transformers  pipeline  method. Chunking is enabled by setting chunk_length_s=30 when instantiating the pipeline. With chunking enabled, the pipeline  can be run with batched inference. It can also be extended to predict sequence level timestamps by passing return_timestamps=True: Refer to the blog post ASR Chunking for more details on the chunking algorithm. Fine-Tuning The pre-trained Whisper model demonstrates a strong ability to generalise to different datasets and domains. However,  its predictive capabilities can be improved further for certain languages and tasks through fine-tuning. The blog  post Fine-Tune Whisper with 🤗 Transformers provides a step-by-step  guide to fine-tuning the Whisper model with as little as 5 hours of labelled data. Evaluated Use The primary intended users of these models are AI researchers studying robustness, generalization, capabilities, biases, and constraints of the current model. However, Whisper is also potentially quite useful as an ASR solution for developers, especially for English speech recognition. We recognize that once models are released, it is impossible to restrict access to only “intended” uses or to draw reasonable guidelines around what is or is not research. The models are primarily trained and evaluated on ASR and speech translation to English tasks. They show strong ASR results in ~10 languages. They may exhibit additional capabilities, particularly if fine-tuned on certain tasks like voice activity detection, speaker classification, or speaker diarization but have not been robustly evaluated in these areas. We strongly recommend that users perform robust evaluations of the models in a particular context and domain before deploying them. In particular, we caution against using Whisper models to transcribe recordings of individuals taken without their consent or purporting to use these models for any kind of subjective classification. We recommend against use in high-risk domains like decision-making contexts, where flaws in accuracy can lead to pronounced flaws in outcomes. The models are intended to transcribe and translate speech, use of the model for classification is not only not evaluated but also not appropriate, particularly to infer human attributes. Training Data The models are trained on 680,000 hours of audio and the corresponding transcripts collected from the internet. 65% of this data (or 438,000 hours) represents English-language audio and matched English transcripts, roughly 18% (or 126,000 hours) represents non-English audio and English transcripts, while the final 17% (or 117,000 hours) represents non-English audio and the corresponding transcript. This non-English data represents 98 different languages.  As discussed in the accompanying paper, we see that performance on transcription in a given language is directly correlated with the amount of training data we employ in that language. Performance and Limitations Our studies show that, over many existing ASR systems, the models exhibit improved robustness to accents, background noise, technical language, as well as zero shot translation from multiple languages into English; and that accuracy on speech recognition and translation is near the state-of-the-art level.  However, because the models are trained in a weakly supervised manner using large-scale noisy data, the predictions may include texts that are not actually spoken in the audio input (i.e. hallucination). We hypothesize that this happens because, given their general knowledge of language, the models combine trying to predict the next word in audio with trying to transcribe the audio itself. Our models perform unevenly across languages, and we observe lower accuracy on low-resource and/or low-discoverability languages or languages where we have less training data. The models also exhibit disparate performance on different accents and dialects of particular languages, which may include higher word error rate across speakers of different genders, races, ages, or other demographic criteria. Our full evaluation results are presented in the paper accompanying this release.  In addition, the sequence-to-sequence architecture of the model makes it prone to generating repetitive texts, which can be mitigated to some degree by beam search and temperature scheduling but not perfectly. Further analysis on these limitations are provided in the paper. It is likely that this behavior and hallucinations may be worse on lower-resource and/or lower-discoverability languages. Broader Implications We anticipate that Whisper models’ transcription capabilities may be used for improving accessibility tools. While Whisper models cannot be used for real-time transcription out of the box – their speed and size suggest that others may be able to build applications on top of them that allow for near-real-time speech recognition and translation. The real value of beneficial applications built on top of Whisper models suggests that the disparate performance of these models may have real economic implications. There are also potential dual use concerns that come with releasing Whisper. While we hope the technology will be used primarily for beneficial purposes, making ASR technology more accessible could enable more actors to build capable surveillance technologies or scale up existing surveillance efforts, as the speed and accuracy allow for affordable automatic transcription and translation of large volumes of audio communication. Moreover, these models may have some capabilities to recognize specific individuals out of the box, which in turn presents safety concerns related both to dual use and disparate performance. In practice, we expect that the cost of transcription is not the limiting factor of scaling up surveillance projects. BibTeX entry and citation info bibtex @misc{radford2022whisper,   doi = {10.48550/ARXIV.2212.04356},   url = {},   author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},   title = {Robust Speech Recognition via Large-Scale Weak Supervision},   publisher = {arXiv},   year = {2022},   copyright = {arXiv.org perpetual, non-exclusive license} }",
        "markdown_text": "---\nlanguage:\n- en\n- zh\n- de\n- es\n- ru\n- ko\n- fr\n- ja\n- pt\n- tr\n- pl\n- ca\n- nl\n- ar\n- sv\n- it\n- id\n- hi\n- fi\n- vi\n- he\n- uk\n- el\n- ms\n- cs\n- ro\n- da\n- hu\n- ta\n- false\n- th\n- ur\n- hr\n- bg\n- lt\n- la\n- mi\n- ml\n- cy\n- sk\n- te\n- fa\n- lv\n- bn\n- sr\n- az\n- sl\n- kn\n- et\n- mk\n- br\n- eu\n- is\n- hy\n- ne\n- mn\n- bs\n- kk\n- sq\n- sw\n- gl\n- mr\n- pa\n- si\n- km\n- sn\n- yo\n- so\n- af\n- oc\n- ka\n- be\n- tg\n- sd\n- gu\n- am\n- yi\n- lo\n- uz\n- fo\n- ht\n- ps\n- tk\n- nn\n- mt\n- sa\n- lb\n- my\n- bo\n- tl\n- mg\n- as\n- tt\n- haw\n- ln\n- ha\n- ba\n- jw\n- su\nlicense: apache-2.0\ntags:\n- audio\n- automatic-speech-recognition\n- hf-asr-leaderboard\nwidget:\n- example_title: Librispeech sample 1\n  src: https://cdn-media.huggingface.co/speech_samples/sample1.flac\n- example_title: Librispeech sample 2\n  src: https://cdn-media.huggingface.co/speech_samples/sample2.flac\npipeline_tag: automatic-speech-recognition\n---\n\n# Whisper\n\nWhisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. Trained on 680k hours \nof labelled data, Whisper models demonstrate a strong ability to generalise to many datasets and domains **without** the need \nfor fine-tuning.\n\nWhisper was proposed in the paper [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) \nby Alec Radford et al. from OpenAI. The original code repository can be found [here](https://github.com/openai/whisper).\n\nCompared to the Whisper large model, the large-v2 model is trained for 2.5x more epochs with added regularization \nfor improved performance.\n\n**Disclaimer**: Content for this model card has partly been written by the Hugging Face team, and parts of it were \ncopied and pasted from the original model card.\n\n## Model details\n\nWhisper is a Transformer based encoder-decoder model, also referred to as a _sequence-to-sequence_ model. \nIt was trained on 680k hours of labelled speech data annotated using large-scale weak supervision. \n\nThe models were trained on either English-only data or multilingual data. The English-only models were trained \non the task of speech recognition. The multilingual models were trained on both speech recognition and speech \ntranslation. For speech recognition, the model predicts transcriptions in the *same* language as the audio. \nFor speech translation, the model predicts transcriptions to a *different* language to the audio.\n\nWhisper checkpoints come in five configurations of varying model sizes.\nThe smallest four are trained on either English-only or multilingual data.\nThe largest checkpoints are multilingual only. All ten of the pre-trained checkpoints \nare available on the [Hugging Face Hub](https://huggingface.co/models?search=openai/whisper). The \ncheckpoints are summarised in the following table with links to the models on the Hub:\n\n| Size     | Parameters | English-only                                         | Multilingual                                        |\n|----------|------------|------------------------------------------------------|-----------------------------------------------------|\n| tiny     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny)     |\n| base     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)     |\n| small    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)    |\n| medium   | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium)   |\n| large    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)    |\n| large-v2 | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large-v2) |\n\n# Usage\n\nTo transcribe audio samples, the model has to be used alongside a [`WhisperProcessor`](https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperProcessor).\n\nThe `WhisperProcessor` is used to:\n1. Pre-process the audio inputs (converting them to log-Mel spectrograms for the model)\n2. Post-process the model outputs (converting them from tokens to text)\n\nThe model is informed of which task to perform (transcription or translation) by passing the appropriate \"context tokens\". These context tokens \nare a sequence of tokens that are given to the decoder at the start of the decoding process, and take the following order:\n1. The transcription always starts with the `<|startoftranscript|>` token\n2. The second token is the language token (e.g. `<|en|>` for English)\n3. The third token is the \"task token\". It can take one of two values: `<|transcribe|>` for speech recognition or `<|translate|>` for speech translation\n4. In addition, a `<|notimestamps|>` token is added if the model should not include timestamp prediction\n\nThus, a typical sequence of context tokens might look as follows:\n```\n<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>\n```\nWhich tells the model to decode in English, under the task of speech recognition, and not to predict timestamps.\n\nThese tokens can either be forced or un-forced. If they are forced, the model is made to predict each token at \neach position. This allows one to control the output language and task for the Whisper model. If they are un-forced, \nthe Whisper model will automatically predict the output langauge and task itself.\n\nThe context tokens can be set accordingly:\n\n```python\nmodel.config.forced_decoder_ids = WhisperProcessor.get_decoder_prompt_ids(language=\"english\", task=\"transcribe\")\n```\n\nWhich forces the model to predict in English under the task of speech recognition.\n\n## Transcription\n\n### English to English \nIn this example, the context tokens are 'unforced', meaning the model automatically predicts the output language\n(English) and task (transcribe).\n\n```python\n>>> from transformers import WhisperProcessor, WhisperForConditionalGeneration\n>>> from datasets import load_dataset\n\n>>> # load model and processor\n>>> processor = WhisperProcessor.from_pretrained(\"openai/whisper-large-v2\")\n>>> model = WhisperForConditionalGeneration.from_pretrained(\"openai/whisper-large-v2\")\n>>> model.config.forced_decoder_ids = None\n\n>>> # load dummy dataset and read audio files\n>>> ds = load_dataset(\"hf-internal-testing/librispeech_asr_dummy\", \"clean\", split=\"validation\")\n>>> sample = ds[0][\"audio\"]\n>>> input_features = processor(sample[\"array\"], sampling_rate=sample[\"sampling_rate\"], return_tensors=\"pt\").input_features \n\n>>> # generate token ids\n>>> predicted_ids = model.generate(input_features)\n>>> # decode token ids to text\n>>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)\n['<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.<|endoftext|>']\n\n>>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)\n[' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.']\n```\nThe context tokens can be removed from the start of the transcription by setting `skip_special_tokens=True`.\n\n### French to French \nThe following example demonstrates French to French transcription by setting the decoder ids appropriately. \n\n```python\n>>> from transformers import WhisperProcessor, WhisperForConditionalGeneration\n>>> from datasets import Audio, load_dataset\n\n>>> # load model and processor\n>>> processor = WhisperProcessor.from_pretrained(\"openai/whisper-large-v2\")\n>>> model = WhisperForConditionalGeneration.from_pretrained(\"openai/whisper-large-v2\")\n>>> forced_decoder_ids = processor.get_decoder_prompt_ids(language=\"french\", task=\"transcribe\")\n\n>>> # load streaming dataset and read first audio sample\n>>> ds = load_dataset(\"common_voice\", \"fr\", split=\"test\", streaming=True)\n>>> ds = ds.cast_column(\"audio\", Audio(sampling_rate=16_000))\n>>> input_speech = next(iter(ds))[\"audio\"]\n>>> input_features = processor(input_speech[\"array\"], sampling_rate=input_speech[\"sampling_rate\"], return_tensors=\"pt\").input_features\n\n>>> # generate token ids\n>>> predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)\n>>> # decode token ids to text\n>>> transcription = processor.batch_decode(predicted_ids)\n['<|startoftranscript|><|fr|><|transcribe|><|notimestamps|> Un vrai travail intéressant va enfin être mené sur ce sujet.<|endoftext|>']\n\n>>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)\n[' Un vrai travail intéressant va enfin être mené sur ce sujet.']\n```\n\n## Translation \nSetting the task to \"translate\" forces the Whisper model to perform speech translation.\n\n### French to English\n\n```python\n>>> from transformers import WhisperProcessor, WhisperForConditionalGeneration\n>>> from datasets import Audio, load_dataset\n\n>>> # load model and processor\n>>> processor = WhisperProcessor.from_pretrained(\"openai/whisper-large-v2\")\n>>> model = WhisperForConditionalGeneration.from_pretrained(\"openai/whisper-large-v2\")\n>>> forced_decoder_ids = processor.get_decoder_prompt_ids(language=\"french\", task=\"translate\")\n\n>>> # load streaming dataset and read first audio sample\n>>> ds = load_dataset(\"common_voice\", \"fr\", split=\"test\", streaming=True)\n>>> ds = ds.cast_column(\"audio\", Audio(sampling_rate=16_000))\n>>> input_speech = next(iter(ds))[\"audio\"]\n>>> input_features = processor(input_speech[\"array\"], sampling_rate=input_speech[\"sampling_rate\"], return_tensors=\"pt\").input_features\n\n>>> # generate token ids\n>>> predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)\n>>> # decode token ids to text\n>>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)\n[' A very interesting work, we will finally be given on this subject.']\n```\n\n## Evaluation\n\nThis code snippet shows how to evaluate Whisper Large on [LibriSpeech test-clean](https://huggingface.co/datasets/librispeech_asr):\n \n```python\n>>> from datasets import load_dataset\n>>> from transformers import WhisperForConditionalGeneration, WhisperProcessor\n>>> import torch\n>>> from evaluate import load\n\n>>> librispeech_test_clean = load_dataset(\"librispeech_asr\", \"clean\", split=\"test\")\n\n>>> processor = WhisperProcessor.from_pretrained(\"openai/whisper-large-v2\")\n>>> model = WhisperForConditionalGeneration.from_pretrained(\"openai/whisper-large-v2\").to(\"cuda\")\n\n>>> def map_to_pred(batch):\n>>>     audio = batch[\"audio\"]\n>>>     input_features = processor(audio[\"array\"], sampling_rate=audio[\"sampling_rate\"], return_tensors=\"pt\").input_features\n>>>     batch[\"reference\"] = processor.tokenizer._normalize(batch['text'])\n>>> \n>>>     with torch.no_grad():\n>>>         predicted_ids = model.generate(input_features.to(\"cuda\"))[0]\n>>>     transcription = processor.decode(predicted_ids)\n>>>     batch[\"prediction\"] = processor.tokenizer._normalize(transcription)\n>>>     return batch\n\n>>> result = librispeech_test_clean.map(map_to_pred)\n\n>>> wer = load(\"wer\")\n>>> print(100 * wer.compute(references=result[\"reference\"], predictions=result[\"prediction\"]))\n3.0003583080317572\n```\n\n## Long-Form Transcription\n\nThe Whisper model is intrinsically designed to work on audio samples of up to 30s in duration. However, by using a chunking \nalgorithm, it can be used to transcribe audio samples of up to arbitrary length. This is possible through Transformers \n[`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline) \nmethod. Chunking is enabled by setting `chunk_length_s=30` when instantiating the pipeline. With chunking enabled, the pipeline \ncan be run with batched inference. It can also be extended to predict sequence level timestamps by passing `return_timestamps=True`:\n\n```python\n>>> import torch\n>>> from transformers import pipeline\n>>> from datasets import load_dataset\n\n>>> device = \"cuda:0\" if torch.cuda.is_available() else \"cpu\"\n\n>>> pipe = pipeline(\n>>>   \"automatic-speech-recognition\",\n>>>   model=\"openai/whisper-large-v2\",\n>>>   chunk_length_s=30,\n>>>   device=device,\n>>> )\n\n>>> ds = load_dataset(\"hf-internal-testing/librispeech_asr_dummy\", \"clean\", split=\"validation\")\n>>> sample = ds[0][\"audio\"]\n\n>>> prediction = pipe(sample.copy(), batch_size=8)[\"text\"]\n\" Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.\"\n\n>>> # we can also return timestamps for the predictions\n>>> prediction = pipe(sample.copy(), batch_size=8, return_timestamps=True)[\"chunks\"]\n[{'text': ' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.',\n  'timestamp': (0.0, 5.44)}]\n```\n\nRefer to the blog post [ASR Chunking](https://huggingface.co/blog/asr-chunking) for more details on the chunking algorithm.\n\n## Fine-Tuning\n\nThe pre-trained Whisper model demonstrates a strong ability to generalise to different datasets and domains. However, \nits predictive capabilities can be improved further for certain languages and tasks through *fine-tuning*. The blog \npost [Fine-Tune Whisper with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper) provides a step-by-step \nguide to fine-tuning the Whisper model with as little as 5 hours of labelled data.\n\n### Evaluated Use\n\nThe primary intended users of these models are AI researchers studying robustness, generalization, capabilities, biases, and constraints of the current model. However, Whisper is also potentially quite useful as an ASR solution for developers, especially for English speech recognition. We recognize that once models are released, it is impossible to restrict access to only “intended” uses or to draw reasonable guidelines around what is or is not research.\n\nThe models are primarily trained and evaluated on ASR and speech translation to English tasks. They show strong ASR results in ~10 languages. They may exhibit additional capabilities, particularly if fine-tuned on certain tasks like voice activity detection, speaker classification, or speaker diarization but have not been robustly evaluated in these areas. We strongly recommend that users perform robust evaluations of the models in a particular context and domain before deploying them.\n\nIn particular, we caution against using Whisper models to transcribe recordings of individuals taken without their consent or purporting to use these models for any kind of subjective classification. We recommend against use in high-risk domains like decision-making contexts, where flaws in accuracy can lead to pronounced flaws in outcomes. The models are intended to transcribe and translate speech, use of the model for classification is not only not evaluated but also not appropriate, particularly to infer human attributes.\n\n\n## Training Data\n\nThe models are trained on 680,000 hours of audio and the corresponding transcripts collected from the internet. 65% of this data (or 438,000 hours) represents English-language audio and matched English transcripts, roughly 18% (or 126,000 hours) represents non-English audio and English transcripts, while the final 17% (or 117,000 hours) represents non-English audio and the corresponding transcript. This non-English data represents 98 different languages. \n\nAs discussed in [the accompanying paper](https://cdn.openai.com/papers/whisper.pdf), we see that performance on transcription in a given language is directly correlated with the amount of training data we employ in that language.\n\n\n## Performance and Limitations\n\nOur studies show that, over many existing ASR systems, the models exhibit improved robustness to accents, background noise, technical language, as well as zero shot translation from multiple languages into English; and that accuracy on speech recognition and translation is near the state-of-the-art level. \n\nHowever, because the models are trained in a weakly supervised manner using large-scale noisy data, the predictions may include texts that are not actually spoken in the audio input (i.e. hallucination). We hypothesize that this happens because, given their general knowledge of language, the models combine trying to predict the next word in audio with trying to transcribe the audio itself.\n\nOur models perform unevenly across languages, and we observe lower accuracy on low-resource and/or low-discoverability languages or languages where we have less training data. The models also exhibit disparate performance on different accents and dialects of particular languages, which may include higher word error rate across speakers of different genders, races, ages, or other demographic criteria. Our full evaluation results are presented in [the paper accompanying this release](https://cdn.openai.com/papers/whisper.pdf). \n\nIn addition, the sequence-to-sequence architecture of the model makes it prone to generating repetitive texts, which can be mitigated to some degree by beam search and temperature scheduling but not perfectly. Further analysis on these limitations are provided in [the paper](https://cdn.openai.com/papers/whisper.pdf). It is likely that this behavior and hallucinations may be worse on lower-resource and/or lower-discoverability languages.\n\n\n## Broader Implications\n\nWe anticipate that Whisper models’ transcription capabilities may be used for improving accessibility tools. While Whisper models cannot be used for real-time transcription out of the box – their speed and size suggest that others may be able to build applications on top of them that allow for near-real-time speech recognition and translation. The real value of beneficial applications built on top of Whisper models suggests that the disparate performance of these models may have real economic implications.\n\nThere are also potential dual use concerns that come with releasing Whisper. While we hope the technology will be used primarily for beneficial purposes, making ASR technology more accessible could enable more actors to build capable surveillance technologies or scale up existing surveillance efforts, as the speed and accuracy allow for affordable automatic transcription and translation of large volumes of audio communication. Moreover, these models may have some capabilities to recognize specific individuals out of the box, which in turn presents safety concerns related both to dual use and disparate performance. In practice, we expect that the cost of transcription is not the limiting factor of scaling up surveillance projects.\n\n\n### BibTeX entry and citation info\n```bibtex\n@misc{radford2022whisper,\n  doi = {10.48550/ARXIV.2212.04356},\n  url = {https://arxiv.org/abs/2212.04356},\n  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},\n  title = {Robust Speech Recognition via Large-Scale Weak Supervision},\n  publisher = {arXiv},\n  year = {2022},\n  copyright = {arXiv.org perpetual, non-exclusive license}\n}\n```\n",
        "llm_extraction": " these models can also be used by developers to build ASR systems for various applications. \n\nDocument: language: - en - zh - de - es - ru - ko - fr - ja - pt - tr - pl - ca - nl - ar - sv - it - id - hi - fi - vi - he - uk - el - ms - cs - ro - da - hu - ta - false - th - ur - hr - bg - lt - la - mi - ml - cy - sk - te - fa - lv - bn - sr - az - sl - kn - et - mk - br - eu - is - hy - ne - mn - bs - kk - sq - sw - gl - mr - pa - si - km - sn - yo - so - af - oc - ka - be - tk - nn - mt - sa - lb - my - bo - tl - mg - as - tt - haw - ln - ha - ba - jw - su license: apache-2.0 tags: - audio - automatic-speech-recognition - hf-asr-leaderboard widget: - example_title: Librispeech sample 1   src:  - example_title: Librispeech sample 2   src:  pipeline_tag: automatic-speech-recognition  Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. Trained on 680k hours of labelled data, Whisper models demonstrate a strong ability to generalise to many datasets and domains without the need for fine-tuning. Whisper was proposed in the paper Robust Speech Recognition via Large-Scale Weak Supervision by Alec Radford et al. from OpenAI. The original code repository can be found here. Compared to the Whisper large model, the large-v2 model is trained for 2.5x more epochs with added regularization for improved performance. Disclaimer: Content for this model card has partly been written by the Hugging Face team, and parts of it were copied and pasted from the original model card. Model details Whisper is a Transformer based encoder-decoder model, also referred to as a sequence-to-sequence model. It",
        "truncation": 1,
        "extraction_version": "v_1",
        "papers_with_code": "successful",
        "performance": [
            {
                "id": "b59d6f0e-ae83-47bd-9b0e-2ec9f493b7cf",
                "best_rank": 1.0,
                "metrics": {
                    "Word Error Rate (WER)": "9.4%"
                },
                "methodology": "Whisper (Large v2)",
                "uses_additional_data": true,
                "paper": "robust-speech-recognition-via-large-scale-1",
                "best_metric": "Word Error Rate (WER)",
                "evaluated_on": "2022-12-06",
                "evaluation": "speech-recognition-on-common-voice-english",
                "benchmark_details": {
                    "id": "speech-recognition-on-common-voice-english",
                    "task": "speech-recognition",
                    "dataset": "common-voice-english",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "b7417174-d4f8-43a0-8f45-25478dc926cc",
                "best_rank": NaN,
                "metrics": {
                    "Test WER": "13.9%"
                },
                "methodology": "Whisper (Large v2)",
                "uses_additional_data": true,
                "paper": "robust-speech-recognition-via-large-scale-1",
                "best_metric": null,
                "evaluated_on": "2022-12-06",
                "evaluation": "speech-recognition-on-common-voice-french",
                "benchmark_details": {
                    "id": "speech-recognition-on-common-voice-french",
                    "task": "speech-recognition",
                    "dataset": "common-voice-french",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "1a8366e4-e79e-45ad-ab00-d475765ae1fa",
                "best_rank": NaN,
                "metrics": {
                    "Test WER": "6.4%"
                },
                "methodology": "Whisper (Large v2)",
                "uses_additional_data": true,
                "paper": "robust-speech-recognition-via-large-scale-1",
                "best_metric": null,
                "evaluated_on": "2022-12-06",
                "evaluation": "speech-recognition-on-common-voice-german",
                "benchmark_details": {
                    "id": "speech-recognition-on-common-voice-german",
                    "task": "speech-recognition",
                    "dataset": "common-voice-german",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "95ed0281-9a19-410e-9019-7a7f08dc9426",
                "best_rank": 1.0,
                "metrics": {
                    "Test WER": "7.1%"
                },
                "methodology": "Whisper (Large v2)",
                "uses_additional_data": true,
                "paper": "robust-speech-recognition-via-large-scale-1",
                "best_metric": "Test WER",
                "evaluated_on": "2022-12-06",
                "evaluation": "speech-recognition-on-common-voice-italian",
                "benchmark_details": {
                    "id": "speech-recognition-on-common-voice-italian",
                    "task": "speech-recognition",
                    "dataset": "common-voice-italian",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "c6a548f8-c19e-4905-9ddc-d7eae9aa8428",
                "best_rank": NaN,
                "metrics": {
                    "Test WER": "5.6%"
                },
                "methodology": "Whisper (Large v2)",
                "uses_additional_data": true,
                "paper": "robust-speech-recognition-via-large-scale-1",
                "best_metric": null,
                "evaluated_on": "2022-12-06",
                "evaluation": "speech-recognition-on-common-voice-spanish",
                "benchmark_details": {
                    "id": "speech-recognition-on-common-voice-spanish",
                    "task": "speech-recognition",
                    "dataset": "common-voice-spanish",
                    "description": "",
                    "mirror_url": null
                }
            }
        ],
        "model_usage": {
            "llm_input": {
                "Whisper/ Model details": "Whisper is a Transformer based encoder-decoder model, also referred to as a sequence-to-sequence model. \nIt was trained on 680k hours of labelled speech data annotated using large-scale weak supervision. \nThe models were trained on either English-only data or multilingual data. The English-only models were trained \non the task of speech recognition. The multilingual models were trained on both speech recognition and speech \ntranslation. For speech recognition, the model predicts transcriptions in the same language as the audio. \nFor speech translation, the model predicts transcriptions to a different language to the audio.\nWhisper checkpoints come in five configurations of varying model sizes.\nThe smallest four are trained on either English-only or multilingual data.\nThe largest checkpoints are multilingual only. All ten of the pre-trained checkpoints \nare available on the Hugging Face Hub. The \ncheckpoints are summarised in the following table with links to the models on the Hub:\n| large-v2 | 1550 M     | x                                                    | ✓ |",
                "Transcription/ English to English": "In this example, the context tokens are 'unforced', meaning the model automatically predicts the output language\n(English) and task (transcribe).\nThe context tokens can be removed from the start of the transcription by setting skip_special_tokens=True.",
                "Transcription/ French to French": "The following example demonstrates French to French transcription by setting the decoder ids appropriately. ",
                "Translation / French to English": "",
                "Usage/ Long-Form Transcription": "The Whisper model is intrinsically designed to work on audio samples of up to 30s in duration. However, by using a chunking \nalgorithm, it can be used to transcribe audio samples of up to arbitrary length. This is possible through Transformers \npipeline \nmethod. Chunking is enabled by setting chunk_length_s=30 when instantiating the pipeline. With chunking enabled, the pipeline \ncan be run with batched inference. It can also be extended to predict sequence level timestamps by passing return_timestamps=True:\nRefer to the blog post ASR Chunking for more details on the chunking algorithm.",
                "Usage/ Training Data": "The models are trained on 680,000 hours of audio and the corresponding transcripts collected from the internet. 65% of this data (or 438,000 hours) represents English-language audio and matched English transcripts, roughly 18% (or 126,000 hours) represents non-English audio and English transcripts, while the final 17% (or 117,000 hours) represents non-English audio and the corresponding transcript. This non-English data represents 98 different languages. \nAs discussed in the accompanying paper, we see that performance on transcription in a given language is directly correlated with the amount of training data we employ in that language.",
                "Usage/ Performance and Limitations": "Our studies show that, over many existing ASR systems, the models exhibit improved robustness to accents, background noise, technical language, as well as zero shot translation from multiple languages into English; and that accuracy on speech recognition and translation is near the state-of-the-art level. \nHowever, because the models are trained in a weakly supervised manner using large-scale noisy data, the predictions may include texts that are not actually spoken in the audio input (i.e. hallucination). We hypothesize that this happens because, given their general knowledge of language, the models combine trying to predict the next word in audio with trying to transcribe the audio itself.\nOur models perform unevenly across languages, and we observe lower accuracy on low-resource and/or low-discoverability languages or languages where we have less training data. The models also exhibit disparate performance on different accents and dialects of particular languages, which may include higher word error rate across speakers of different genders, races, ages, or other demographic criteria. Our full evaluation results are presented in the paper accompanying this release. \nIn addition, the sequence-to-sequence architecture of the model makes it prone to generating repetitive texts, which can be mitigated to some degree by beam search and temperature scheduling but not perfectly. Further analysis on these limitations are provided in the paper. It is likely that this behavior and hallucinations may be worse on lower-resource and/or lower-discoverability languages.",
                "Broader Implications/ BibTeX entry and citation info": "@misc{radford2022whisper,\n  doi = {10.48550/ARXIV.2212.04356},\n  url = {https://arxiv.org/abs/2212.04356},\n  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},\n  title = {Robust Speech Recognition via Large-Scale Weak Supervision},\n  publisher = {arXiv},\n  year = {2022},\n  copyright = {arXiv.org perpetual, non-exclusive license}\n}"
            },
            "usage": {
                "Fine-Tuning/ Evaluated Use": "The primary intended users of these models are AI researchers studying robustness, generalization, capabilities, biases, and constraints of the current model. However, Whisper is also potentially quite useful as an ASR solution for developers, especially for English speech recognition. We recognize that once models are released, it is impossible to restrict access to only “intended” uses or to draw reasonable guidelines around what is or is not research.\nThe models are primarily trained and evaluated on ASR and speech translation to English tasks. They show strong ASR results in ~10 languages. They may exhibit additional capabilities, particularly if fine-tuned on certain tasks like voice activity detection, speaker classification, or speaker diarization but have not been robustly evaluated in these areas. We strongly recommend that users perform robust evaluations of the models in a particular context and domain before deploying them.\nIn particular, we caution against using Whisper models to transcribe recordings of individuals taken without their consent or purporting to use these models for any kind of subjective classification. We recommend against use in high-risk domains like decision-making contexts, where flaws in accuracy can lead to pronounced flaws in outcomes. The models are intended to transcribe and translate speech, use of the model for classification is not only not evaluated but also not appropriate, particularly to infer human attributes."
            },
            "model_function": [
                {
                    "function_info": {
                        "return": "str",
                        "function_name": "automatic_speech_recognition",
                        "variables": [
                            {
                                "name": "input",
                                "type": "str",
                                "default": "hello"
                            }
                        ]
                    }
                }
            ]
        }
    },
    "stabilityai/stable-diffusion-2-1": {
        "model_name": "stable-diffusion-2-1",
        "org": "stabilityai",
        "model_info": {
            "id": "stabilityai/stable-diffusion-2-1",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 1341822,
            "likes": 3531,
            "library_name": "diffusers",
            "tags": [
                "diffusers",
                "safetensors",
                "stable-diffusion",
                "text-to-image",
                "arxiv:2112.10752",
                "arxiv:2202.00512",
                "arxiv:1910.09700",
                "license:openrail++",
                "endpoints_compatible",
                "has_space",
                "diffusers:StableDiffusionPipeline",
                "region:us"
            ],
            "pipeline_tag": "text-to-image",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "638f7ae36c25af4071044105",
            "createdAt": "2022-12-06T17:24:51.000Z",
            "modelId": "stabilityai/stable-diffusion-2-1"
        },
        "card_to_dict": {
            "license": "openrail++",
            "tags": [
                "stable-diffusion",
                "text-to-image"
            ],
            "pinned": true
        },
        "relevant_websites": [
            "https://github.com/Stability-AI/stablediffusion",
            "https://huggingface.co/stabilityai/stable-diffusion-2",
            "https://github.com/Stability-AI/stablediffusion",
            "https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.ckpt",
            "https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL",
            "https://arxiv.org/abs/2112.10752",
            "https://github.com/mlfoundations/open_clip",
            "https://github.com/Stability-AI",
            "https://github.com/huggingface/diffusers",
            "https://github.com/facebookresearch/xformers",
            "https://huggingface.co/dalle-mini/dalle-mini",
            "https://laion.ai/blog/laion-5b",
            "https://laion.ai/blog/laion-5b",
            "https://openreview.net/forum?id=M3Y74vmsMcY",
            "https://arxiv.org/abs/2202.00512",
            "https://laion.ai/blog/laion-5b",
            "https://github.com/LAION-AI/CLIP-based-NSFW-Detector",
            "https://github.com/christophschuhmann/improved-aesthetic-predictor",
            "https://arxiv.org/abs/2202.00512",
            "https://github.com/isl-org/MiDaS",
            "https://github.com/saic-mdal/lama",
            "https://huggingface.co/runwayml/stable-diffusion-inpainting",
            "https://arxiv.org/abs/2112.10752",
            "https://mlco2.github.io/impact#compute",
            "https://arxiv.org/abs/1910.09700",
            "https://github.com/CompVis/stable-diffusion/blob/main/Stable_Diffusion_v1_Model_Card.md",
            "https://huggingface.co/dalle-mini/dalle-mini"
        ],
        "text": "license: openrail++ tags: - stable-diffusion - text-to-image pinned: true  Stable Diffusion v2-1 Model Card This model card focuses on the model associated with the Stable Diffusion v2-1 model, codebase available here. This stable-diffusion-2-1 model is fine-tuned from stable-diffusion-2 (768-v-ema.ckpt) with an additional 55k steps on the same dataset (with punsafe=0.1), and then fine-tuned for another 155k extra steps with punsafe=0.98.  Use it with the stablediffusion repository: download the v2-1_768-ema-pruned.ckpt here. Use it with 🧨 diffusers  Model Details  Developed by: Robin Rombach, Patrick Esser Model type: Diffusion-based text-to-image generation model Language(s): English License: CreativeML Open RAIL++-M License Model Description: This is a model that can be used to generate and modify images based on text prompts. It is a Latent Diffusion Model that uses a fixed, pretrained text encoder (OpenCLIP-ViT/H). Resources for more information: GitHub Repository.  Cite as: @InProceedings{Rombach_2022_CVPR,       author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\\\"orn},       title     = {High-Resolution Image Synthesis With Latent Diffusion Models},       booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},       month     = {June},       year      = {2022},       pages     = {10684-10695}   }   Examples Using the 🤗's Diffusers library to run Stable Diffusion 2 in a simple and efficient manner. bash pip install diffusers transformers accelerate scipy safetensors Running the pipeline (if you don't swap the scheduler it will run with the default DDIM, in this example we are swapping it to DPMSolverMultistepScheduler): Notes: - Despite not being a dependency, we highly recommend you to install xformers for memory efficient attention (better performance) - If you have low GPU RAM available, make sure to add a pipe.enable_attention_slicing() after sending it to cuda for less VRAM usage (to the cost of speed) Uses Direct Use The model is intended for research purposes only. Possible research areas and tasks include  Safe deployment of models which have the potential to generate harmful content. Probing and understanding the limitations and biases of generative models. Generation of artworks and use in design and other artistic processes. Applications in educational or creative tools. Research on generative models.  Excluded uses are described below. ### Misuse, Malicious Use, and Out-of-Scope Use Note: This section is originally taken from the DALLE-MINI model card, was used for Stable Diffusion v1, but applies in the same way to Stable Diffusion v2. The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes. Out-of-Scope Use The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model. Misuse and Malicious Use Using the model to generate content that is cruel to individuals is a misuse of this model. This includes, but is not limited to:  Generating demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc. Intentionally promoting or propagating discriminatory content or harmful stereotypes. Impersonating individuals without their consent. Sexual content without consent of the people who might see it. Mis- and disinformation Representations of egregious violence and gore Sharing of copyrighted or licensed material in violation of its terms of use. Sharing content that is an alteration of copyrighted or licensed material in violation of its terms of use.  Limitations and Bias Limitations  The model does not achieve perfect photorealism The model cannot render legible text The model does not perform well on more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere” Faces and people in general may not be generated properly. The model was trained mainly with English captions and will not work as well in other languages. The autoencoding part of the model is lossy The model was trained on a subset of the large-scale dataset   LAION-5B, which contains adult, violent and sexual content. To partially mitigate this, we have filtered the dataset using LAION's NFSW detector (see Training section).  Bias While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases.  Stable Diffusion was primarily trained on subsets of LAION-2B(en),  which consists of images that are limited to English descriptions.  Texts and images from communities and cultures that use other languages are likely to be insufficiently accounted for.  This affects the overall output of the model, as white and western cultures are often set as the default. Further, the  ability of the model to generate content with non-English prompts is significantly worse than with English-language prompts. Stable Diffusion v2 mirrors and exacerbates biases to such a degree that viewer discretion must be advised irrespective of the input or its intent. Training Training Data The model developers used the following dataset for training the model:  LAION-5B and subsets (details below). The training data is further filtered using LAION's NSFW detector, with a \"p_unsafe\" score of 0.1 (conservative). For more details, please refer to LAION-5B's NeurIPS 2022 paper and reviewer discussions on the topic.  Training Procedure Stable Diffusion v2 is a latent diffusion model which combines an autoencoder with a diffusion model that is trained in the latent space of the autoencoder. During training,   Images are encoded through an encoder, which turns images into latent representations. The autoencoder uses a relative downsampling factor of 8 and maps images of shape H x W x 3 to latents of shape H/f x W/f x 4 Text prompts are encoded through the OpenCLIP-ViT/H text-encoder. The output of the text encoder is fed into the UNet backbone of the latent diffusion model via cross-attention. The loss is a reconstruction objective between the noise that was added to the latent and the prediction made by the UNet. We also use the so-called v-objective, see .  We currently provide the following checkpoints:  512-base-ema.ckpt: 550k steps at resolution 256x256 on a subset of LAION-5B filtered for explicit pornographic material, using the LAION-NSFW classifier with punsafe=0.1 and an aesthetic score >= 4.5.   850k steps at resolution 512x512 on the same dataset with resolution >= 512x512. 768-v-ema.ckpt: Resumed from 512-base-ema.ckpt and trained for 150k steps using a v-objective on the same dataset. Resumed for another 140k steps on a 768x768 subset of our dataset. 512-depth-ema.ckpt: Resumed from 512-base-ema.ckpt and finetuned for 200k steps. Added an extra input channel to process the (relative) depth prediction produced by MiDaS (dpt_hybrid) which is used as an additional conditioning. The additional input channels of the U-Net which process this extra information were zero-initialized. 512-inpainting-ema.ckpt: Resumed from 512-base-ema.ckpt and trained for another 200k steps. Follows the mask-generation strategy presented in LAMA which, in combination with the latent VAE representations of the masked image, are used as an additional conditioning. The additional input channels of the U-Net which process this extra information were zero-initialized. The same strategy was used to train the 1.5-inpainting checkpoint.  x4-upscaling-ema.ckpt: Trained for 1.25M steps on a 10M subset of LAION containing images >2048x2048. The model was trained on crops of size 512x512 and is a text-guided latent upscaling diffusion model. In addition to the textual input, it receives a noise_level as an input parameter, which can be used to add noise to the low-resolution input according to a predefined diffusion schedule.    Hardware: 32 x 8 x A100 GPUs  Optimizer: AdamW Gradient Accumulations: 1 Batch: 32 x 8 x 2 x 4 = 2048 Learning rate: warmup to 0.0001 for 10,000 steps and then kept constant  Evaluation Results Evaluations with different classifier-free guidance scales (1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0) and 50 steps DDIM sampling steps show the relative improvements of the checkpoints:   Evaluated using 50 DDIM steps and 10000 random prompts from the COCO2017 validation set, evaluated at 512x512 resolution.  Not optimized for FID scores. Environmental Impact Stable Diffusion v1 Estimated Emissions Based on that information, we estimate the following CO2 emissions using the Machine Learning Impact calculator presented in Lacoste et al. (2019). The hardware, runtime, cloud provider, and compute region were utilized to estimate the carbon impact.  Hardware Type: A100 PCIe 40GB Hours used: 200000 Cloud Provider: AWS Compute Region: US-east Carbon Emitted (Power consumption x Time x Carbon produced based on location of power grid): 15000 kg CO2 eq.  Citation @InProceedings{Rombach_2022_CVPR,     author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\\\"orn},     title     = {High-Resolution Image Synthesis With Latent Diffusion Models},     booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},     month     = {June},     year      = {2022},     pages     = {10684-10695} }  This model card was written by: Robin Rombach, Patrick Esser and David Ha and is based on the Stable Diffusion v1 and DALL-E Mini model card.",
        "markdown_text": "---\nlicense: openrail++\ntags:\n- stable-diffusion\n- text-to-image\npinned: true\n---\n\n# Stable Diffusion v2-1 Model Card\nThis model card focuses on the model associated with the Stable Diffusion v2-1 model, codebase available [here](https://github.com/Stability-AI/stablediffusion).\n\nThis `stable-diffusion-2-1` model is fine-tuned from [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) (`768-v-ema.ckpt`) with an additional 55k steps on the same dataset (with `punsafe=0.1`), and then fine-tuned for another 155k extra steps with `punsafe=0.98`.\n\n- Use it with the [`stablediffusion`](https://github.com/Stability-AI/stablediffusion) repository: download the `v2-1_768-ema-pruned.ckpt` [here](https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.ckpt).\n- Use it with 🧨 [`diffusers`](#examples)\n\n## Model Details\n- **Developed by:** Robin Rombach, Patrick Esser\n- **Model type:** Diffusion-based text-to-image generation model\n- **Language(s):** English\n- **License:** [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL)\n- **Model Description:** This is a model that can be used to generate and modify images based on text prompts. It is a [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) that uses a fixed, pretrained text encoder ([OpenCLIP-ViT/H](https://github.com/mlfoundations/open_clip)).\n- **Resources for more information:** [GitHub Repository](https://github.com/Stability-AI/).\n- **Cite as:**\n\n      @InProceedings{Rombach_2022_CVPR,\n          author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\\\"orn},\n          title     = {High-Resolution Image Synthesis With Latent Diffusion Models},\n          booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},\n          month     = {June},\n          year      = {2022},\n          pages     = {10684-10695}\n      }\n\n\n## Examples\n\nUsing the [🤗's Diffusers library](https://github.com/huggingface/diffusers) to run Stable Diffusion 2 in a simple and efficient manner.\n\n```bash\npip install diffusers transformers accelerate scipy safetensors\n```\nRunning the pipeline (if you don't swap the scheduler it will run with the default DDIM, in this example we are swapping it to DPMSolverMultistepScheduler):\n\n```python\nimport torch\nfrom diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler\n\nmodel_id = \"stabilityai/stable-diffusion-2-1\"\n\n# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead\npipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)\npipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)\npipe = pipe.to(\"cuda\")\n\nprompt = \"a photo of an astronaut riding a horse on mars\"\nimage = pipe(prompt).images[0]\n    \nimage.save(\"astronaut_rides_horse.png\")\n```\n\n**Notes**:\n- Despite not being a dependency, we highly recommend you to install [xformers](https://github.com/facebookresearch/xformers) for memory efficient attention (better performance)\n- If you have low GPU RAM available, make sure to add a `pipe.enable_attention_slicing()` after sending it to `cuda` for less VRAM usage (to the cost of speed)\n\n\n# Uses\n\n## Direct Use \nThe model is intended for research purposes only. Possible research areas and tasks include\n\n- Safe deployment of models which have the potential to generate harmful content.\n- Probing and understanding the limitations and biases of generative models.\n- Generation of artworks and use in design and other artistic processes.\n- Applications in educational or creative tools.\n- Research on generative models.\n\nExcluded uses are described below.\n\n ### Misuse, Malicious Use, and Out-of-Scope Use\n_Note: This section is originally taken from the [DALLE-MINI model card](https://huggingface.co/dalle-mini/dalle-mini), was used for Stable Diffusion v1, but applies in the same way to Stable Diffusion v2_.\n\nThe model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.\n\n#### Out-of-Scope Use\nThe model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.\n\n#### Misuse and Malicious Use\nUsing the model to generate content that is cruel to individuals is a misuse of this model. This includes, but is not limited to:\n\n- Generating demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc.\n- Intentionally promoting or propagating discriminatory content or harmful stereotypes.\n- Impersonating individuals without their consent.\n- Sexual content without consent of the people who might see it.\n- Mis- and disinformation\n- Representations of egregious violence and gore\n- Sharing of copyrighted or licensed material in violation of its terms of use.\n- Sharing content that is an alteration of copyrighted or licensed material in violation of its terms of use.\n\n## Limitations and Bias\n\n### Limitations\n\n- The model does not achieve perfect photorealism\n- The model cannot render legible text\n- The model does not perform well on more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”\n- Faces and people in general may not be generated properly.\n- The model was trained mainly with English captions and will not work as well in other languages.\n- The autoencoding part of the model is lossy\n- The model was trained on a subset of the large-scale dataset\n  [LAION-5B](https://laion.ai/blog/laion-5b/), which contains adult, violent and sexual content. To partially mitigate this, we have filtered the dataset using LAION's NFSW detector (see Training section).\n\n### Bias\nWhile the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases. \nStable Diffusion was primarily trained on subsets of [LAION-2B(en)](https://laion.ai/blog/laion-5b/), \nwhich consists of images that are limited to English descriptions. \nTexts and images from communities and cultures that use other languages are likely to be insufficiently accounted for. \nThis affects the overall output of the model, as white and western cultures are often set as the default. Further, the \nability of the model to generate content with non-English prompts is significantly worse than with English-language prompts.\nStable Diffusion v2 mirrors and exacerbates biases to such a degree that viewer discretion must be advised irrespective of the input or its intent.\n\n\n## Training\n\n**Training Data**\nThe model developers used the following dataset for training the model:\n\n- LAION-5B and subsets (details below). The training data is further filtered using LAION's NSFW detector, with a \"p_unsafe\" score of 0.1 (conservative). For more details, please refer to LAION-5B's [NeurIPS 2022](https://openreview.net/forum?id=M3Y74vmsMcY) paper and reviewer discussions on the topic.\n\n**Training Procedure**\nStable Diffusion v2 is a latent diffusion model which combines an autoencoder with a diffusion model that is trained in the latent space of the autoencoder. During training, \n\n- Images are encoded through an encoder, which turns images into latent representations. The autoencoder uses a relative downsampling factor of 8 and maps images of shape H x W x 3 to latents of shape H/f x W/f x 4\n- Text prompts are encoded through the OpenCLIP-ViT/H text-encoder.\n- The output of the text encoder is fed into the UNet backbone of the latent diffusion model via cross-attention.\n- The loss is a reconstruction objective between the noise that was added to the latent and the prediction made by the UNet. We also use the so-called _v-objective_, see https://arxiv.org/abs/2202.00512.\n\nWe currently provide the following checkpoints:\n\n- `512-base-ema.ckpt`: 550k steps at resolution `256x256` on a subset of [LAION-5B](https://laion.ai/blog/laion-5b/) filtered for explicit pornographic material, using the [LAION-NSFW classifier](https://github.com/LAION-AI/CLIP-based-NSFW-Detector) with `punsafe=0.1` and an [aesthetic score](https://github.com/christophschuhmann/improved-aesthetic-predictor) >= `4.5`.\n  850k steps at resolution `512x512` on the same dataset with resolution `>= 512x512`.\n- `768-v-ema.ckpt`: Resumed from `512-base-ema.ckpt` and trained for 150k steps using a [v-objective](https://arxiv.org/abs/2202.00512) on the same dataset. Resumed for another 140k steps on a `768x768` subset of our dataset.\n- `512-depth-ema.ckpt`: Resumed from `512-base-ema.ckpt` and finetuned for 200k steps. Added an extra input channel to process the (relative) depth prediction produced by [MiDaS](https://github.com/isl-org/MiDaS) (`dpt_hybrid`) which is used as an additional conditioning.\nThe additional input channels of the U-Net which process this extra information were zero-initialized.\n- `512-inpainting-ema.ckpt`: Resumed from `512-base-ema.ckpt` and trained for another 200k steps. Follows the mask-generation strategy presented in [LAMA](https://github.com/saic-mdal/lama) which, in combination with the latent VAE representations of the masked image, are used as an additional conditioning.\nThe additional input channels of the U-Net which process this extra information were zero-initialized. The same strategy was used to train the [1.5-inpainting checkpoint](https://huggingface.co/runwayml/stable-diffusion-inpainting).\n- `x4-upscaling-ema.ckpt`: Trained for 1.25M steps on a 10M subset of LAION containing images `>2048x2048`. The model was trained on crops of size `512x512` and is a text-guided [latent upscaling diffusion model](https://arxiv.org/abs/2112.10752).\nIn addition to the textual input, it receives a `noise_level` as an input parameter, which can be used to add noise to the low-resolution input according to a [predefined diffusion schedule](configs/stable-diffusion/x4-upscaling.yaml). \n\n- **Hardware:** 32 x 8 x A100 GPUs\n- **Optimizer:** AdamW\n- **Gradient Accumulations**: 1\n- **Batch:** 32 x 8 x 2 x 4 = 2048\n- **Learning rate:** warmup to 0.0001 for 10,000 steps and then kept constant\n\n## Evaluation Results \nEvaluations with different classifier-free guidance scales (1.5, 2.0, 3.0, 4.0,\n5.0, 6.0, 7.0, 8.0) and 50 steps DDIM sampling steps show the relative improvements of the checkpoints:\n\n![pareto](model-variants.jpg) \n\nEvaluated using 50 DDIM steps and 10000 random prompts from the COCO2017 validation set, evaluated at 512x512 resolution.  Not optimized for FID scores.\n\n## Environmental Impact\n\n**Stable Diffusion v1** **Estimated Emissions**\nBased on that information, we estimate the following CO2 emissions using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). The hardware, runtime, cloud provider, and compute region were utilized to estimate the carbon impact.\n\n- **Hardware Type:** A100 PCIe 40GB\n- **Hours used:** 200000\n- **Cloud Provider:** AWS\n- **Compute Region:** US-east\n- **Carbon Emitted (Power consumption x Time x Carbon produced based on location of power grid):** 15000 kg CO2 eq.\n\n## Citation\n    @InProceedings{Rombach_2022_CVPR,\n        author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\\\"orn},\n        title     = {High-Resolution Image Synthesis With Latent Diffusion Models},\n        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},\n        month     = {June},\n        year      = {2022},\n        pages     = {10684-10695}\n    }\n\n*This model card was written by: Robin Rombach, Patrick Esser and David Ha and is based on the [Stable Diffusion v1](https://github.com/CompVis/stable-diffusion/blob/main/Stable_Diffusion_v1_Model_Card.md) and [DALL-E Mini model card](https://huggingface.co/dalle-mini/dalle-mini).*\n",
        "llm_extraction": ". The model was fine-tuned from stable-diffusion-2 (768-v-ema.ckpt) with an additional 55k steps on the same dataset (with punsafe=0.1), and then fine-tuned for another 155k extra steps with punsafe=0.98.  The model was trained using the DDIM optimizer with a batch size of 128 and a learning rate of 1e-4.  The model was trained on a single GPU.  The model was trained for 100 epochs.  The model was trained on a single GPU.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 100 epochs.  The model was trained for 1",
        "truncation": 1,
        "extraction_version": "v_1",
        "papers_with_code": "Performance not found on papers with code",
        "model_usage": {
            "llm_input": {
                "Stable Diffusion v2-1 Model Card/ Model Details": [
                    "**Developed by:** Robin Rombach, Patrick Esser",
                    "**Model type:** Diffusion-based text-to-image generation model",
                    "**Language(s):** English",
                    "**License:** [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL)",
                    "**Model Description:** This is a model that can be used to generate and modify images based on text prompts. It is a [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) that uses a fixed, pretrained text encoder ([OpenCLIP-ViT/H](https://github.com/mlfoundations/open_clip)).",
                    "**Resources for more information:** [GitHub Repository](https://github.com/Stability-AI/).",
                    "**Cite as:**",
                    "@InProceedings{Rombach_2022_CVPR,\n    author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\\\"orn},\n    title     = {High-Resolution Image Synthesis With Latent Diffusion Models},\n    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},\n    month     = {June},\n    year      = {2022},\n    pages     = {10684-10695}\n}\n"
                ],
                "Stable Diffusion v2-1 Model Card/ Examples": "Using the 🤗's Diffusers library to run Stable Diffusion 2 in a simple and efficient manner.\npip install diffusers transformers accelerate scipy safetensors\nRunning the pipeline (if you don't swap the scheduler it will run with the default DDIM, in this example we are swapping it to DPMSolverMultistepScheduler):\nNotes:\n['Despite not being a dependency, we highly recommend you to install xformers for memory efficient attention (better performance)', 'If you have low GPU RAM available, make sure to add a pipe.enable_attention_slicing() after sending it to cuda for less VRAM usage (to the cost of speed)']",
                "Limitations and Bias/ Limitations": [
                    "The model does not achieve perfect photorealism",
                    "The model cannot render legible text",
                    "The model does not perform well on more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”",
                    "Faces and people in general may not be generated properly.",
                    "The model was trained mainly with English captions and will not work as well in other languages.",
                    "The autoencoding part of the model is lossy",
                    "The model was trained on a subset of the large-scale dataset\n[LAION-5B](https://laion.ai/blog/laion-5b/), which contains adult, violent and sexual content. To partially mitigate this, we have filtered the dataset using LAION's NFSW detector (see Training section)."
                ],
                "Limitations and Bias/ Bias": "While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases. \nStable Diffusion was primarily trained on subsets of LAION-2B(en), \nwhich consists of images that are limited to English descriptions. \nTexts and images from communities and cultures that use other languages are likely to be insufficiently accounted for. \nThis affects the overall output of the model, as white and western cultures are often set as the default. Further, the \nability of the model to generate content with non-English prompts is significantly worse than with English-language prompts.\nStable Diffusion v2 mirrors and exacerbates biases to such a degree that viewer discretion must be advised irrespective of the input or its intent.",
                "Uses/ Training": "Training Data\nThe model developers used the following dataset for training the model:\n['LAION-5B and subsets (details below). The training data is further filtered using LAION\\'s NSFW detector, with a \"p_unsafe\" score of 0.1 (conservative). For more details, please refer to LAION-5B\\'s NeurIPS 2022 paper and reviewer discussions on the topic.']\nTraining Procedure\nStable Diffusion v2 is a latent diffusion model which combines an autoencoder with a diffusion model that is trained in the latent space of the autoencoder. During training, \n['Images are encoded through an encoder, which turns images into latent representations. The autoencoder uses a relative downsampling factor of 8 and maps images of shape H x W x 3 to latents of shape H/f x W/f x 4', 'Text prompts are encoded through the OpenCLIP-ViT/H text-encoder.', 'The output of the text encoder is fed into the UNet backbone of the latent diffusion model via cross-attention.', 'The loss is a reconstruction objective between the noise that was added to the latent and the prediction made by the UNet. We also use the so-called v-objective, see https://arxiv.org/abs/2202.00512.']\nWe currently provide the following checkpoints:\n['512-base-ema.ckpt: 550k steps at resolution 256x256 on a subset of LAION-5B filtered for explicit pornographic material, using the LAION-NSFW classifier with punsafe=0.1 and an aesthetic score >= 4.5.\\n850k steps at resolution 512x512 on the same dataset with resolution >= 512x512.', '768-v-ema.ckpt: Resumed from 512-base-ema.ckpt and trained for 150k steps using a v-objective on the same dataset. Resumed for another 140k steps on a 768x768 subset of our dataset.', '512-depth-ema.ckpt: Resumed from 512-base-ema.ckpt and finetuned for 200k steps. Added an extra input channel to process the (relative) depth prediction produced by MiDaS (dpt_hybrid) which is used as an additional conditioning.\\nThe additional input channels of the U-Net which process this extra information were zero-initialized.', '512-inpainting-ema.ckpt: Resumed from 512-base-ema.ckpt and trained for another 200k steps. Follows the mask-generation strategy presented in LAMA which, in combination with the latent VAE representations of the masked image, are used as an additional conditioning.\\nThe additional input channels of the U-Net which process this extra information were zero-initialized. The same strategy was used to train the 1.5-inpainting checkpoint.', 'x4-upscaling-ema.ckpt: Trained for 1.25M steps on a 10M subset of LAION containing images >2048x2048. The model was trained on crops of size 512x512 and is a text-guided latent upscaling diffusion model.\\nIn addition to the textual input, it receives a noise_level as an input parameter, which can be used to add noise to the low-resolution input according to a predefined diffusion schedule. ', 'Hardware: 32 x 8 x A100 GPUs', 'Optimizer: AdamW', 'Gradient Accumulations: 1', 'Batch: 32 x 8 x 2 x 4 = 2048', 'Learning rate: warmup to 0.0001 for 10,000 steps and then kept constant']",
                "Uses/ Environmental Impact": "Stable Diffusion v1 Estimated Emissions\nBased on that information, we estimate the following CO2 emissions using the Machine Learning Impact calculator presented in Lacoste et al. (2019). The hardware, runtime, cloud provider, and compute region were utilized to estimate the carbon impact.\n['Hardware Type: A100 PCIe 40GB', 'Hours used: 200000', 'Cloud Provider: AWS', 'Compute Region: US-east', 'Carbon Emitted (Power consumption x Time x Carbon produced based on location of power grid): 15000 kg CO2 eq.']",
                "Uses/ Citation": "@InProceedings{Rombach_2022_CVPR,\n    author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\\\"orn},\n    title     = {High-Resolution Image Synthesis With Latent Diffusion Models},\n    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},\n    month     = {June},\n    year      = {2022},\n    pages     = {10684-10695}\n}\nThis model card was written by: Robin Rombach, Patrick Esser and David Ha and is based on the Stable Diffusion v1 and DALL-E Mini model card."
            },
            "usage": {
                "Misuse, Malicious Use, and Out-of-Scope Use/ Out-of-Scope Use": "The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.",
                "Misuse, Malicious Use, and Out-of-Scope Use/ Misuse and Malicious Use": "Using the model to generate content that is cruel to individuals is a misuse of this model. This includes, but is not limited to:\n['Generating demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc.', 'Intentionally promoting or propagating discriminatory content or harmful stereotypes.', 'Impersonating individuals without their consent.', 'Sexual content without consent of the people who might see it.', 'Mis- and disinformation', 'Representations of egregious violence and gore', 'Sharing of copyrighted or licensed material in violation of its terms of use.', 'Sharing content that is an alteration of copyrighted or licensed material in violation of its terms of use.']"
            },
            "model_function": []
        }
    },
    "deepseek-ai/deepseek-coder-6.7b-instruct": {
        "model_name": "deepseek-coder-6.7b-instruct",
        "org": "deepseek-ai",
        "model_info": {
            "modelId": "deepseek-ai/deepseek-coder-6.7b-instruct",
            "sha": null,
            "lastModified": null,
            "tags": [
                "transformers",
                "pytorch",
                "safetensors",
                "llama",
                "text-generation",
                "conversational",
                "license:other",
                "autotrain_compatible",
                "endpoints_compatible",
                "has_space",
                "text-generation-inference",
                "region:us"
            ],
            "pipeline_tag": "text-generation",
            "siblings": [],
            "private": false,
            "author": null,
            "config": null,
            "securityStatus": null,
            "_id": "653e3b904a52f10eaf646821",
            "id": "deepseek-ai/deepseek-coder-6.7b-instruct",
            "likes": 284,
            "downloads": 44188,
            "library_name": "transformers",
            "createdAt": "2023-10-29T11:01:36.000Z"
        },
        "card_to_dict": {
            "license": "other",
            "license_name": "deepseek",
            "license_link": "LICENSE"
        },
        "papers_with_code": "Performance not found on papers with code",
        "relevant_websites": [
            "https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/pictures/logo.png?raw=true",
            "https://www.deepseek.com/\">[🏠Homepage]</a",
            "https://coder.deepseek.com",
            "https://discord.gg/Tc7c45Zzu5\">[Discord]</a",
            "https://github.com/guoday/assert/blob/main/QR.png?raw=true\">[Wechat(微信)]</a",
            "https://deepseek.com",
            "https://github.com/deepseek-ai/deepseek-coder",
            "https://coder.deepseek.com",
            "https://github.com/deepseek-ai/deepseek-coder/blob/main/LICENSE-MODEL"
        ],
        "text": "[🤖 Chat with DeepSeek Coder]  |     1. Introduction of Deepseek Coder Deepseek Coder is composed of a series of code language models, each trained from scratch on 2T tokens, with a composition of 87% code and 13% natural language in both English and Chinese. We provide various sizes of the code model, ranging from 1B to 33B versions. Each model is pre-trained on project-level code corpus by employing a window size of 16K and a extra fill-in-the-blank task, to support  project-level code completion and infilling. For coding capabilities, Deepseek Coder achieves state-of-the-art performance among open-source code models on multiple programming languages and various benchmarks.    Massive Training Data: Trained from scratch fon 2T tokens, including 87% code and 13% linguistic data in both English and Chinese languages.   Highly Flexible & Scalable: Offered in model sizes of 1.3B, 5.7B, 6.7B, and 33B, enabling users to choose the setup most suitable for their requirements.   Superior Model Performance: State-of-the-art performance among publicly available code models on HumanEval, MultiPL-E, MBPP, DS-1000, and APPS benchmarks.   Advanced Code Completion Capabilities: A window size of 16K and a fill-in-the-blank task, supporting project-level code completion and infilling tasks.   2. Model Summary deepseek-coder-6.7b-instruct is a 6.7B parameter model initialized from deepseek-coder-6.7b-base and fine-tuned on 2B tokens of instruction data. - Home Page: DeepSeek - Repository: deepseek-ai/deepseek-coder - Chat With DeepSeek Coder: DeepSeek-Coder 3. How to Use Here give some examples of how to use our model. Chat Model Inference 4. License This code repository is licensed under the MIT License. The use of DeepSeek Coder models is subject to the Model License. DeepSeek Coder supports commercial use. See the LICENSE-MODEL for more details. 5. Contact If you have any questions, please raise an issue or contact us at agi_code@deepseek.com.",
        "markdown_text": "\n<p align=\"center\">\n<img width=\"1000px\" alt=\"DeepSeek Coder\" src=\"https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/pictures/logo.png?raw=true\">\n</p>\n<p align=\"center\"><a href=\"https://www.deepseek.com/\">[🏠Homepage]</a>  |  <a href=\"https://coder.deepseek.com/\">[🤖 Chat with DeepSeek Coder]</a>  |  <a href=\"https://discord.gg/Tc7c45Zzu5\">[Discord]</a>  |  <a href=\"https://github.com/guoday/assert/blob/main/QR.png?raw=true\">[Wechat(微信)]</a> </p>\n<hr>\n\n\n\n\n### 1. Introduction of Deepseek Coder\n\nDeepseek Coder is composed of a series of code language models, each trained from scratch on 2T tokens, with a composition of 87% code and 13% natural language in both English and Chinese. We provide various sizes of the code model, ranging from 1B to 33B versions. Each model is pre-trained on project-level code corpus by employing a window size of 16K and a extra fill-in-the-blank task, to support  project-level code completion and infilling. For coding capabilities, Deepseek Coder achieves state-of-the-art performance among open-source code models on multiple programming languages and various benchmarks. \n\n- **Massive Training Data**: Trained from scratch fon 2T tokens, including 87% code and 13% linguistic data in both English and Chinese languages.\n  \n- **Highly Flexible & Scalable**: Offered in model sizes of 1.3B, 5.7B, 6.7B, and 33B, enabling users to choose the setup most suitable for their requirements.\n  \n- **Superior Model Performance**: State-of-the-art performance among publicly available code models on HumanEval, MultiPL-E, MBPP, DS-1000, and APPS benchmarks.\n  \n- **Advanced Code Completion Capabilities**: A window size of 16K and a fill-in-the-blank task, supporting project-level code completion and infilling tasks.\n\n \n  \n### 2. Model Summary\ndeepseek-coder-6.7b-instruct is a 6.7B parameter model initialized from deepseek-coder-6.7b-base and fine-tuned on 2B tokens of instruction data.\n- **Home Page:** [DeepSeek](https://deepseek.com/)\n- **Repository:** [deepseek-ai/deepseek-coder](https://github.com/deepseek-ai/deepseek-coder)\n- **Chat With DeepSeek Coder:** [DeepSeek-Coder](https://coder.deepseek.com/)\n\n\n### 3. How to Use\nHere give some examples of how to use our model.\n#### Chat Model Inference\n```python\nfrom transformers import AutoTokenizer, AutoModelForCausalLM\ntokenizer = AutoTokenizer.from_pretrained(\"deepseek-ai/deepseek-coder-6.7b-instruct\", trust_remote_code=True)\nmodel = AutoModelForCausalLM.from_pretrained(\"deepseek-ai/deepseek-coder-6.7b-instruct\", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()\nmessages=[\n    { 'role': 'user', 'content': \"write a quick sort algorithm in python.\"}\n]\ninputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors=\"pt\").to(model.device)\n# tokenizer.eos_token_id is the id of <|EOT|> token\noutputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)\nprint(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))\n```\n\n### 4. License\nThis code repository is licensed under the MIT License. The use of DeepSeek Coder models is subject to the Model License. DeepSeek Coder supports commercial use.\n\nSee the [LICENSE-MODEL](https://github.com/deepseek-ai/deepseek-coder/blob/main/LICENSE-MODEL) for more details.\n\n### 5. Contact\n\nIf you have any questions, please raise an issue or contact us at [agi_code@deepseek.com](mailto:agi_code@deepseek.com).\n\n",
        "model_usage": {
            "llm_input": {
                "Introduction of Deepseek Coder": "Deepseek Coder is composed of a series of code language models, each trained from scratch on 2T tokens, with a composition of 87% code and 13% natural language in both English and Chinese. We provide various sizes of the code model, ranging from 1B to 33B versions. Each model is pre-trained on project-level code corpus by employing a window size of 16K and a extra fill-in-the-blank task, to support  project-level code completion and infilling. For coding capabilities, Deepseek Coder achieves state-of-the-art performance among open-source code models on multiple programming languages and various benchmarks. \n['Massive Training Data: Trained from scratch fon 2T tokens, including 87% code and 13% linguistic data in both English and Chinese languages.', 'Highly Flexible & Scalable: Offered in model sizes of 1.3B, 5.7B, 6.7B, and 33B, enabling users to choose the setup most suitable for their requirements.', 'Superior Model Performance: State-of-the-art performance among publicly available code models on HumanEval, MultiPL-E, MBPP, DS-1000, and APPS benchmarks.', 'Advanced Code Completion Capabilities: A window size of 16K and a fill-in-the-blank task, supporting project-level code completion and infilling tasks.']",
                "Model Summary": "deepseek-coder-6.7b-instruct is a 6.7B parameter model initialized from deepseek-coder-6.7b-base and fine-tuned on 2B tokens of instruction data.\n['Home Page: DeepSeek', 'Repository: deepseek-ai/deepseek-coder', 'Chat With DeepSeek Coder: DeepSeek-Coder']",
                "How to Use/ Chat Model Inference": "",
                "License": "This code repository is licensed under the MIT License. The use of DeepSeek Coder models is subject to the Model License. DeepSeek Coder supports commercial use.\nSee the LICENSE-MODEL for more details.",
                "Contact": "If you have any questions, please raise an issue or contact us at agi_code@deepseek.com."
            },
            "usage": {
                "How to Use/ Chat Model Inference": "```\nfrom transformers import AutoTokenizer, AutoModelForCausalLM\ntokenizer = AutoTokenizer.from_pretrained(\"deepseek-ai/deepseek-coder-6.7b-instruct\", trust_remote_code=True)\nmodel = AutoModelForCausalLM.from_pretrained(\"deepseek-ai/deepseek-coder-6.7b-instruct\", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()\nmessages=[\n    { 'role': 'user', 'content': \"write a quick sort algorithm in python.\"}\n]\ninputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors=\"pt\").to(model.device)\ntokenizer.eos_token_id is the id of <|EOT|> token\noutputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)\nprint(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))\n```"
            },
            "model_function": [
                {
                    "code": "from transformers import AutoTokenizer, AutoModelForCausalLM\nimport torch\n\ndef generate_code(model_name=\"deepseek-ai/deepseek-coder-6.7b-instruct\", \n                  messages=[{'role': 'user', 'content': \"write a quick sort algorithm in python.\"}],\n                  max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1):\n    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)\n    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()\n    \n    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors=\"pt\").to(model.device)\n    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, \n                             top_p=top_p, num_return_sequences=num_return_sequences, eos_token_id=tokenizer.eos_token_id)\n    \n    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)",
                    "function_info": {
                        "return": null,
                        "function_name": "generate_response",
                        "variables": [
                            {
                                "name": "model_name",
                                "type": "str",
                                "default": "deepseek-ai/deepseek-coder-6.7b-instruct"
                            },
                            {
                                "name": "messages",
                                "type": "list",
                                "default": [
                                    {
                                        "role": "user",
                                        "content": "write a quick sort algorithm in python."
                                    }
                                ]
                            },
                            {
                                "name": "max_new_tokens",
                                "type": "int",
                                "default": 512
                            },
                            {
                                "name": "do_sample",
                                "type": null,
                                "default": null
                            },
                            {
                                "name": "top_k",
                                "type": "int",
                                "default": 50
                            },
                            {
                                "name": "top_p",
                                "type": "float",
                                "default": 0.95
                            },
                            {
                                "name": "num_return_sequences",
                                "type": "int",
                                "default": 1
                            }
                        ]
                    }
                }
            ]
        }
    },
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "model_name": "stable-diffusion-xl-base-1.0",
        "org": "stabilityai",
        "model_info": {
            "modelId": "stabilityai/stable-diffusion-xl-base-1.0",
            "sha": null,
            "lastModified": null,
            "tags": [
                "diffusers",
                "onnx",
                "safetensors",
                "text-to-image",
                "stable-diffusion",
                "arxiv:2307.01952",
                "arxiv:2211.01324",
                "arxiv:2108.01073",
                "arxiv:2112.10752",
                "license:openrail++",
                "endpoints_compatible",
                "has_space",
                "diffusers:StableDiffusionXLPipeline",
                "region:us"
            ],
            "pipeline_tag": "text-to-image",
            "siblings": [],
            "private": false,
            "author": null,
            "config": null,
            "securityStatus": null,
            "_id": "64bfcd5ff462a99a04fd1ec8",
            "id": "stabilityai/stable-diffusion-xl-base-1.0",
            "likes": 5008,
            "downloads": 4961615,
            "library_name": "diffusers",
            "createdAt": "2023-07-25T13:25:51.000Z"
        },
        "card_to_dict": {
            "license": "openrail++",
            "tags": [
                "text-to-image",
                "stable-diffusion"
            ]
        },
        "papers_with_code": "Performance not found on papers with code",
        "relevant_websites": [
            "https://arxiv.org/abs/2307.01952",
            "https://arxiv.org/abs/2211.01324",
            "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0",
            "https://arxiv.org/abs/2108.01073",
            "https://github.com/Stability-AI/generative-models",
            "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md",
            "https://arxiv.org/abs/2112.10752",
            "https://github.com/mlfoundations/open_clip",
            "https://github.com/openai/CLIP/tree/main",
            "https://github.com/Stability-AI/generative-models",
            "https://arxiv.org/abs/2307.01952",
            "https://github.com/Stability-AI/generative-models",
            "https://clipdrop.co/stable-diffusion",
            "https://github.com/Stability-AI/generative-models",
            "https://clipdrop.co/stable-diffusion",
            "https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl",
            "https://github.com/huggingface/optimum",
            "https://docs.openvino.ai/latest/index.html",
            "https://onnxruntime.ai",
            "https://huggingface.co/docs/optimum/main/en/intel/inference#stable-diffusion-xl",
            "https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/models#stable-diffusion-xl"
        ],
        "text": "SD-XL 1.0-base Model Card  Model  SDXL consists of an ensemble of experts pipeline for latent diffusion:  In a first step, the base model is used to generate (noisy) latents,  which are then further processed with a refinement model (available here: /) specialized for the final denoising steps. Note that the base model can be used as a standalone module. Alternatively, we can use a two-stage pipeline as follows:  First, the base model is used to generate latents of the desired output size.  In the second step, we use a specialized high-resolution model and apply a technique called SDEdit (, also known as \"img2img\")  to the latents generated in the first step, using the same prompt. This technique is slightly slower than the first one, as it requires more function evaluations. Source code is available at  . Model Description  Developed by: Stability AI Model type: Diffusion-based text-to-image generative model License: CreativeML Open RAIL++-M License Model Description: This is a model that can be used to generate and modify images based on text prompts. It is a Latent Diffusion Model that uses two fixed, pretrained text encoders (OpenCLIP-ViT/G and CLIP-ViT/L). Resources for more information: Check out our GitHub Repository and the SDXL report on arXiv.  Model Sources For research purposes, we recommend our generative-models Github repository (), which implements the most popular diffusion frameworks (both training and inference) and for which new functionalities like distillation will be added over time. Clipdrop provides free SDXL inference.  Repository:  Demo:   Evaluation  The chart above evaluates user preference for SDXL (with and without refinement) over SDXL 0.9 and Stable Diffusion 1.5 and 2.1.  The SDXL base model performs significantly better than the previous variants, and the model combined with the refinement module achieves the best overall performance. 🧨 Diffusers Make sure to upgrade diffusers to >= 0.19.0: pip install diffusers --upgrade In addition make sure to install transformers, safetensors, accelerate as well as the invisible watermark: pip install invisible_watermark transformers accelerate safetensors To just use the base model, you can run: ```py from diffusers import DiffusionPipeline import torch pipe = DiffusionPipeline.from_pretrained(\"stabilityai/stable-diffusion-xl-base-1.0\", torch_dtype=torch.float16, use_safetensors=True, variant=\"fp16\") pipe.to(\"cuda\") if using torch < 2.0 pipe.enable_xformers_memory_efficient_attention() prompt = \"An astronaut riding a green horse\" images = pipe(prompt=prompt).images[0] ``` To use the whole base + refiner pipeline as an ensemble of experts you can run: ```py from diffusers import DiffusionPipeline import torch load both base & refiner base = DiffusionPipeline.from_pretrained(     \"stabilityai/stable-diffusion-xl-base-1.0\", torch_dtype=torch.float16, variant=\"fp16\", use_safetensors=True ) base.to(\"cuda\") refiner = DiffusionPipeline.from_pretrained(     \"stabilityai/stable-diffusion-xl-refiner-1.0\",     text_encoder_2=base.text_encoder_2,     vae=base.vae,     torch_dtype=torch.float16,     use_safetensors=True,     variant=\"fp16\", ) refiner.to(\"cuda\") Define how many steps and what % of steps to be run on each experts (80/20) here n_steps = 40 high_noise_frac = 0.8 prompt = \"A majestic lion jumping from a big stone at night\" run both experts image = base(     prompt=prompt,     num_inference_steps=n_steps,     denoising_end=high_noise_frac,     output_type=\"latent\", ).images image = refiner(     prompt=prompt,     num_inference_steps=n_steps,     denoising_start=high_noise_frac,     image=image, ).images[0] ``` When using torch >= 2.0, you can improve the inference speed by 20-30% with torch.compile. Simple wrap the unet with torch compile before running the pipeline: py pipe.unet = torch.compile(pipe.unet, mode=\"reduce-overhead\", fullgraph=True) If you are limited by GPU VRAM, you can enable cpu offloading by calling pipe.enable_model_cpu_offload instead of .to(\"cuda\"): diff - pipe.to(\"cuda\") + pipe.enable_model_cpu_offload() For more information on how to use Stable Diffusion XL with diffusers, please have a look at the Stable Diffusion XL Docs. Optimum Optimum provides a Stable Diffusion pipeline compatible with both OpenVINO and ONNX Runtime. OpenVINO To install Optimum with the dependencies required for OpenVINO : bash pip install optimum[openvino] To load an OpenVINO model and run inference with OpenVINO Runtime, you need to replace StableDiffusionXLPipeline with Optimum OVStableDiffusionXLPipeline. In case you want to load a PyTorch model and convert it to the OpenVINO format on-the-fly, you can set export=True. ```diff - from diffusers import StableDiffusionXLPipeline + from optimum.intel import OVStableDiffusionXLPipeline model_id = \"stabilityai/stable-diffusion-xl-base-1.0\" - pipeline = StableDiffusionXLPipeline.from_pretrained(model_id) + pipeline = OVStableDiffusionXLPipeline.from_pretrained(model_id) prompt = \"A majestic lion jumping from a big stone at night\" image = pipeline(prompt).images[0] ``` You can find more examples (such as static reshaping and model compilation) in optimum documentation. ONNX To install Optimum with the dependencies required for ONNX Runtime inference : bash pip install optimum[onnxruntime] To load an ONNX model and run inference with ONNX Runtime, you need to replace StableDiffusionXLPipeline with Optimum ORTStableDiffusionXLPipeline. In case you want to load a PyTorch model and convert it to the ONNX format on-the-fly, you can set export=True. ```diff - from diffusers import StableDiffusionXLPipeline + from optimum.onnxruntime import ORTStableDiffusionXLPipeline model_id = \"stabilityai/stable-diffusion-xl-base-1.0\" - pipeline = StableDiffusionXLPipeline.from_pretrained(model_id) + pipeline = ORTStableDiffusionXLPipeline.from_pretrained(model_id) prompt = \"A majestic lion jumping from a big stone at night\" image = pipeline(prompt).images[0] ``` You can find more examples in optimum documentation. Uses Direct Use The model is intended for research purposes only. Possible research areas and tasks include  Generation of artworks and use in design and other artistic processes. Applications in educational or creative tools. Research on generative models. Safe deployment of models which have the potential to generate harmful content. Probing and understanding the limitations and biases of generative models.  Excluded uses are described below. Out-of-Scope Use The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model. Limitations and Bias Limitations  The model does not achieve perfect photorealism The model cannot render legible text The model struggles with more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere” Faces and people in general may not be generated properly. The autoencoding part of the model is lossy.  Bias While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases.",
        "markdown_text": "# SD-XL 1.0-base Model Card\n![row01](01.png)\n\n## Model\n\n![pipeline](pipeline.png)\n\n[SDXL](https://arxiv.org/abs/2307.01952) consists of an [ensemble of experts](https://arxiv.org/abs/2211.01324) pipeline for latent diffusion: \nIn a first step, the base model is used to generate (noisy) latents, \nwhich are then further processed with a refinement model (available here: https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/) specialized for the final denoising steps.\nNote that the base model can be used as a standalone module.\n\nAlternatively, we can use a two-stage pipeline as follows: \nFirst, the base model is used to generate latents of the desired output size. \nIn the second step, we use a specialized high-resolution model and apply a technique called SDEdit (https://arxiv.org/abs/2108.01073, also known as \"img2img\") \nto the latents generated in the first step, using the same prompt. This technique is slightly slower than the first one, as it requires more function evaluations.\n\nSource code is available at https://github.com/Stability-AI/generative-models .\n\n### Model Description\n\n- **Developed by:** Stability AI\n- **Model type:** Diffusion-based text-to-image generative model\n- **License:** [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)\n- **Model Description:** This is a model that can be used to generate and modify images based on text prompts. It is a [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) that uses two fixed, pretrained text encoders ([OpenCLIP-ViT/G](https://github.com/mlfoundations/open_clip) and [CLIP-ViT/L](https://github.com/openai/CLIP/tree/main)).\n- **Resources for more information:** Check out our [GitHub Repository](https://github.com/Stability-AI/generative-models) and the [SDXL report on arXiv](https://arxiv.org/abs/2307.01952).\n\n### Model Sources\n\nFor research purposes, we recommend our `generative-models` Github repository (https://github.com/Stability-AI/generative-models), which implements the most popular diffusion frameworks (both training and inference) and for which new functionalities like distillation will be added over time.\n[Clipdrop](https://clipdrop.co/stable-diffusion) provides free SDXL inference.\n\n- **Repository:** https://github.com/Stability-AI/generative-models\n- **Demo:** https://clipdrop.co/stable-diffusion\n\n\n## Evaluation\n![comparison](comparison.png)\nThe chart above evaluates user preference for SDXL (with and without refinement) over SDXL 0.9 and Stable Diffusion 1.5 and 2.1. \nThe SDXL base model performs significantly better than the previous variants, and the model combined with the refinement module achieves the best overall performance.\n\n\n### 🧨 Diffusers \n\nMake sure to upgrade diffusers to >= 0.19.0:\n```\npip install diffusers --upgrade\n```\n\nIn addition make sure to install `transformers`, `safetensors`, `accelerate` as well as the invisible watermark:\n```\npip install invisible_watermark transformers accelerate safetensors\n```\n\nTo just use the base model, you can run:\n\n```py\nfrom diffusers import DiffusionPipeline\nimport torch\n\npipe = DiffusionPipeline.from_pretrained(\"stabilityai/stable-diffusion-xl-base-1.0\", torch_dtype=torch.float16, use_safetensors=True, variant=\"fp16\")\npipe.to(\"cuda\")\n\n# if using torch < 2.0\n# pipe.enable_xformers_memory_efficient_attention()\n\nprompt = \"An astronaut riding a green horse\"\n\nimages = pipe(prompt=prompt).images[0]\n```\n\nTo use the whole base + refiner pipeline as an ensemble of experts you can run:\n\n```py\nfrom diffusers import DiffusionPipeline\nimport torch\n\n# load both base & refiner\nbase = DiffusionPipeline.from_pretrained(\n    \"stabilityai/stable-diffusion-xl-base-1.0\", torch_dtype=torch.float16, variant=\"fp16\", use_safetensors=True\n)\nbase.to(\"cuda\")\nrefiner = DiffusionPipeline.from_pretrained(\n    \"stabilityai/stable-diffusion-xl-refiner-1.0\",\n    text_encoder_2=base.text_encoder_2,\n    vae=base.vae,\n    torch_dtype=torch.float16,\n    use_safetensors=True,\n    variant=\"fp16\",\n)\nrefiner.to(\"cuda\")\n\n# Define how many steps and what % of steps to be run on each experts (80/20) here\nn_steps = 40\nhigh_noise_frac = 0.8\n\nprompt = \"A majestic lion jumping from a big stone at night\"\n\n# run both experts\nimage = base(\n    prompt=prompt,\n    num_inference_steps=n_steps,\n    denoising_end=high_noise_frac,\n    output_type=\"latent\",\n).images\nimage = refiner(\n    prompt=prompt,\n    num_inference_steps=n_steps,\n    denoising_start=high_noise_frac,\n    image=image,\n).images[0]\n```\n\nWhen using `torch >= 2.0`, you can improve the inference speed by 20-30% with torch.compile. Simple wrap the unet with torch compile before running the pipeline:\n```py\npipe.unet = torch.compile(pipe.unet, mode=\"reduce-overhead\", fullgraph=True)\n```\n\nIf you are limited by GPU VRAM, you can enable *cpu offloading* by calling `pipe.enable_model_cpu_offload`\ninstead of `.to(\"cuda\")`:\n\n```diff\n- pipe.to(\"cuda\")\n+ pipe.enable_model_cpu_offload()\n```\n\nFor more information on how to use Stable Diffusion XL with `diffusers`, please have a look at [the Stable Diffusion XL Docs](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl).\n\n### Optimum\n[Optimum](https://github.com/huggingface/optimum) provides a Stable Diffusion pipeline compatible with both [OpenVINO](https://docs.openvino.ai/latest/index.html) and [ONNX Runtime](https://onnxruntime.ai/).\n\n#### OpenVINO\n\nTo install Optimum with the dependencies required for OpenVINO :\n\n```bash\npip install optimum[openvino]\n```\n\nTo load an OpenVINO model and run inference with OpenVINO Runtime, you need to replace `StableDiffusionXLPipeline` with Optimum `OVStableDiffusionXLPipeline`. In case you want to load a PyTorch model and convert it to the OpenVINO format on-the-fly, you can set `export=True`.\n\n```diff\n- from diffusers import StableDiffusionXLPipeline\n+ from optimum.intel import OVStableDiffusionXLPipeline\n\nmodel_id = \"stabilityai/stable-diffusion-xl-base-1.0\"\n- pipeline = StableDiffusionXLPipeline.from_pretrained(model_id)\n+ pipeline = OVStableDiffusionXLPipeline.from_pretrained(model_id)\nprompt = \"A majestic lion jumping from a big stone at night\"\nimage = pipeline(prompt).images[0]\n```\n\nYou can find more examples (such as static reshaping and model compilation) in optimum [documentation](https://huggingface.co/docs/optimum/main/en/intel/inference#stable-diffusion-xl).\n\n\n#### ONNX\n\nTo install Optimum with the dependencies required for ONNX Runtime inference :\n\n```bash\npip install optimum[onnxruntime]\n```\n\nTo load an ONNX model and run inference with ONNX Runtime, you need to replace `StableDiffusionXLPipeline` with Optimum `ORTStableDiffusionXLPipeline`. In case you want to load a PyTorch model and convert it to the ONNX format on-the-fly, you can set `export=True`.\n\n```diff\n- from diffusers import StableDiffusionXLPipeline\n+ from optimum.onnxruntime import ORTStableDiffusionXLPipeline\n\nmodel_id = \"stabilityai/stable-diffusion-xl-base-1.0\"\n- pipeline = StableDiffusionXLPipeline.from_pretrained(model_id)\n+ pipeline = ORTStableDiffusionXLPipeline.from_pretrained(model_id)\nprompt = \"A majestic lion jumping from a big stone at night\"\nimage = pipeline(prompt).images[0]\n```\n\nYou can find more examples in optimum [documentation](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/models#stable-diffusion-xl).\n\n\n## Uses\n\n### Direct Use\n\nThe model is intended for research purposes only. Possible research areas and tasks include\n\n- Generation of artworks and use in design and other artistic processes.\n- Applications in educational or creative tools.\n- Research on generative models.\n- Safe deployment of models which have the potential to generate harmful content.\n- Probing and understanding the limitations and biases of generative models.\n\nExcluded uses are described below.\n\n### Out-of-Scope Use\n\nThe model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.\n\n## Limitations and Bias\n\n### Limitations\n\n- The model does not achieve perfect photorealism\n- The model cannot render legible text\n- The model struggles with more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”\n- Faces and people in general may not be generated properly.\n- The autoencoding part of the model is lossy.\n\n### Bias\nWhile the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases.\n",
        "model_usage": {
            "llm_input": {
                "SD-XL 1.0-base Model Card": {
                    "Model/ Model Description": [
                        "**Developed by:** Stability AI",
                        "**Model type:** Diffusion-based text-to-image generative model",
                        "**License:** [CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)",
                        "**Model Description:** This is a model that can be used to generate and modify images based on text prompts. It is a [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) that uses two fixed, pretrained text encoders ([OpenCLIP-ViT/G](https://github.com/mlfoundations/open_clip) and [CLIP-ViT/L](https://github.com/openai/CLIP/tree/main)).",
                        "**Resources for more information:** Check out our [GitHub Repository](https://github.com/Stability-AI/generative-models) and the [SDXL report on arXiv](https://arxiv.org/abs/2307.01952)."
                    ],
                    "Model/ Model Sources": "For research purposes, we recommend our generative-models Github repository (https://github.com/Stability-AI/generative-models), which implements the most popular diffusion frameworks (both training and inference) and for which new functionalities like distillation will be added over time.\nClipdrop provides free SDXL inference.\n['Repository: https://github.com/Stability-AI/generative-models', 'Demo: https://clipdrop.co/stable-diffusion']",
                    "Optimum/ OpenVINO": "To install Optimum with the dependencies required for OpenVINO :\npip install optimum[openvino]\nTo load an OpenVINO model and run inference with OpenVINO Runtime, you need to replace StableDiffusionXLPipeline with Optimum OVStableDiffusionXLPipeline. In case you want to load a PyTorch model and convert it to the OpenVINO format on-the-fly, you can set export=True.\n```\n- from diffusers import StableDiffusionXLPipeline\n+ from optimum.intel import OVStableDiffusionXLPipeline\nmodel_id = \"stabilityai/stable-diffusion-xl-base-1.0\"\n- pipeline = StableDiffusionXLPipeline.from_pretrained(model_id)\n+ pipeline = OVStableDiffusionXLPipeline.from_pretrained(model_id)\nprompt = \"A majestic lion jumping from a big stone at night\"\nimage = pipeline(prompt).images[0]\n```\nYou can find more examples (such as static reshaping and model compilation) in optimum documentation.",
                    "Optimum/ ONNX": "To install Optimum with the dependencies required for ONNX Runtime inference :\npip install optimum[onnxruntime]\nTo load an ONNX model and run inference with ONNX Runtime, you need to replace StableDiffusionXLPipeline with Optimum ORTStableDiffusionXLPipeline. In case you want to load a PyTorch model and convert it to the ONNX format on-the-fly, you can set export=True.\n```\n- from diffusers import StableDiffusionXLPipeline\n+ from optimum.onnxruntime import ORTStableDiffusionXLPipeline\nmodel_id = \"stabilityai/stable-diffusion-xl-base-1.0\"\n- pipeline = StableDiffusionXLPipeline.from_pretrained(model_id)\n+ pipeline = ORTStableDiffusionXLPipeline.from_pretrained(model_id)\nprompt = \"A majestic lion jumping from a big stone at night\"\nimage = pipeline(prompt).images[0]\n```\nYou can find more examples in optimum documentation.",
                    "Limitations and Bias/ Limitations": [
                        "The model does not achieve perfect photorealism",
                        "The model cannot render legible text",
                        "The model struggles with more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”",
                        "Faces and people in general may not be generated properly.",
                        "The autoencoding part of the model is lossy."
                    ],
                    "Limitations and Bias/ Bias": "While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases."
                }
            },
            "usage": {
                "Evaluation/ 🧨 Diffusers": "Make sure to upgrade diffusers to >= 0.19.0:\npip install diffusers --upgrade\nIn addition make sure to install transformers, safetensors, accelerate as well as the invisible watermark:\npip install invisible_watermark transformers accelerate safetensors\nTo just use the base model, you can run:\n```\nfrom diffusers import DiffusionPipeline\nimport torch\npipe = DiffusionPipeline.from_pretrained(\"stabilityai/stable-diffusion-xl-base-1.0\", torch_dtype=torch.float16, use_safetensors=True, variant=\"fp16\")\npipe.to(\"cuda\")\nif using torch < 2.0\npipe.enable_xformers_memory_efficient_attention()\nprompt = \"An astronaut riding a green horse\"\nimages = pipe(prompt=prompt).images[0]\n```\nTo use the whole base + refiner pipeline as an ensemble of experts you can run:\n```\nfrom diffusers import DiffusionPipeline\nimport torch\nload both base & refiner\nbase = DiffusionPipeline.from_pretrained(\n    \"stabilityai/stable-diffusion-xl-base-1.0\", torch_dtype=torch.float16, variant=\"fp16\", use_safetensors=True\n)\nbase.to(\"cuda\")\nrefiner = DiffusionPipeline.from_pretrained(\n    \"stabilityai/stable-diffusion-xl-refiner-1.0\",\n    text_encoder_2=base.text_encoder_2,\n    vae=base.vae,\n    torch_dtype=torch.float16,\n    use_safetensors=True,\n    variant=\"fp16\",\n)\nrefiner.to(\"cuda\")\nDefine how many steps and what % of steps to be run on each experts (80/20) here\nn_steps = 40\nhigh_noise_frac = 0.8\nprompt = \"A majestic lion jumping from a big stone at night\"\nrun both experts\nimage = base(\n    prompt=prompt,\n    num_inference_steps=n_steps,\n    denoising_end=high_noise_frac,\n    output_type=\"latent\",\n).images\nimage = refiner(\n    prompt=prompt,\n    num_inference_steps=n_steps,\n    denoising_start=high_noise_frac,\n    image=image,\n).images[0]\n```\nWhen using torch >= 2.0, you can improve the inference speed by 20-30% with torch.compile. Simple wrap the unet with torch compile before running the pipeline:\npipe.unet = torch.compile(pipe.unet, mode=\"reduce-overhead\", fullgraph=True)\nIf you are limited by GPU VRAM, you can enable cpu offloading by calling pipe.enable_model_cpu_offload\ninstead of .to(\"cuda\"):\n- pipe.to(\"cuda\")\n+ pipe.enable_model_cpu_offload()\nFor more information on how to use Stable Diffusion XL with diffusers, please have a look at the Stable Diffusion XL Docs.",
                "Uses/ Direct Use": "The model is intended for research purposes only. Possible research areas and tasks include\n['Generation of artworks and use in design and other artistic processes.', 'Applications in educational or creative tools.', 'Research on generative models.', 'Safe deployment of models which have the potential to generate harmful content.', 'Probing and understanding the limitations and biases of generative models.']\nExcluded uses are described below.",
                "Uses/ Out-of-Scope Use": "The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model."
            },
            "model_function": [
                {
                    "code": "import torch\nfrom diffusers import DiffusionPipeline\n\ndef run_stable_diffusion(prompt=\"An astronaut riding a green horse\", \n                          model_path=\"stabilityai/stable-diffusion-xl-base-1.0\",\n                          use_safetensors=True,\n                          variant=\"fp16\",\n                          torch_dtype=torch.float16,\n                          device=\"cuda\",\n                          n_steps=40,\n                          high_noise_frac=0.8):\n    # Load the base model\n    base = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype, variant=variant, use_safetensors=use_safetensors)\n    \n    if device == \"cuda\":\n        # Enable xformers memory efficient attention for torch < 2.0\n        if int(torch.__version__.split('.')[1]) < 2:\n            base.enable_xformers_memory_efficient_attention()\n        \n        # Compile the unet with torch compile for torch >= 2.0\n        base.unet = torch.compile(base.unet, mode=\"reduce-overhead\", fullgraph=True)\n    \n    base.to(device)\n    \n    # Run the base model\n    image = base(prompt=prompt, num_inference_steps=n_steps, denoising_end=high_noise_frac, output_type=\"latent\").images\n    \n    return image",
                    "function_info": {
                        "return": null,
                        "function_name": "run_model"
                    }
                }
            ]
        }
    },
    "nvidia/parakeet-rnnt-1.1b": {
        "model_name": "parakeet-rnnt-1.1b",
        "org": "nvidia",
        "model_info": {
            "id": "nvidia/parakeet-rnnt-1.1b",
            "author": null,
            "sha": null,
            "last_modified": null,
            "private": false,
            "gated": null,
            "disabled": null,
            "downloads": 5353,
            "likes": 90,
            "library_name": "nemo",
            "tags": [
                "nemo",
                "automatic-speech-recognition",
                "speech",
                "audio",
                "Transducer",
                "FastConformer",
                "Conformer",
                "pytorch",
                "NeMo",
                "hf-asr-leaderboard",
                "en",
                "dataset:librispeech_asr",
                "dataset:fisher_corpus",
                "dataset:Switchboard-1",
                "dataset:WSJ-0",
                "dataset:WSJ-1",
                "dataset:National-Singapore-Corpus-Part-1",
                "dataset:National-Singapore-Corpus-Part-6",
                "dataset:vctk",
                "dataset:voxpopuli",
                "dataset:europarl",
                "dataset:multilingual_librispeech",
                "dataset:mozilla-foundation/common_voice_8_0",
                "dataset:MLCommons/peoples_speech",
                "arxiv:2305.05084",
                "license:cc-by-4.0",
                "model-index",
                "has_space",
                "region:us"
            ],
            "pipeline_tag": "automatic-speech-recognition",
            "mask_token": null,
            "card_data": null,
            "widget_data": null,
            "model_index": null,
            "config": null,
            "transformers_info": null,
            "siblings": null,
            "spaces": null,
            "safetensors": null,
            "lastModified": null,
            "cardData": null,
            "transformersInfo": null,
            "_id": "658cb5dd8cff48d3a45472a7",
            "createdAt": "2023-12-27T23:40:13.000Z",
            "modelId": "nvidia/parakeet-rnnt-1.1b"
        },
        "card_to_dict": {
            "language": [
                "en"
            ],
            "license": "cc-by-4.0",
            "library_name": "nemo",
            "tags": [
                "automatic-speech-recognition",
                "speech",
                "audio",
                "Transducer",
                "FastConformer",
                "Conformer",
                "pytorch",
                "NeMo",
                "hf-asr-leaderboard"
            ],
            "datasets": [
                "librispeech_asr",
                "fisher_corpus",
                "Switchboard-1",
                "WSJ-0",
                "WSJ-1",
                "National-Singapore-Corpus-Part-1",
                "National-Singapore-Corpus-Part-6",
                "vctk",
                "voxpopuli",
                "europarl",
                "multilingual_librispeech",
                "mozilla-foundation/common_voice_8_0",
                "MLCommons/peoples_speech"
            ],
            "metrics": [
                "wer"
            ],
            "widget": [
                {
                    "example_title": "Librispeech sample 1",
                    "src": "https://cdn-media.huggingface.co/speech_samples/sample1.flac"
                },
                {
                    "example_title": "Librispeech sample 2",
                    "src": "https://cdn-media.huggingface.co/speech_samples/sample2.flac"
                }
            ],
            "pipeline_tag": "automatic-speech-recognition",
            "model-index": [
                {
                    "name": "parakeet_rnnt_1.1b",
                    "results": [
                        {
                            "task": {
                                "type": "automatic-speech-recognition",
                                "name": "Automatic Speech Recognition"
                            },
                            "dataset": {
                                "name": "AMI (Meetings test)",
                                "type": "edinburghcstr/ami",
                                "config": "ihm",
                                "split": "test",
                                "args": {
                                    "language": "en"
                                }
                            },
                            "metrics": [
                                {
                                    "type": "wer",
                                    "value": 17.1,
                                    "name": "Test WER"
                                }
                            ]
                        },
                        {
                            "task": {
                                "type": "automatic-speech-recognition",
                                "name": "Automatic Speech Recognition"
                            },
                            "dataset": {
                                "name": "Earnings-22",
                                "type": "revdotcom/earnings22",
                                "split": "test",
                                "args": {
                                    "language": "en"
                                }
                            },
                            "metrics": [
                                {
                                    "type": "wer",
                                    "value": 14.11,
                                    "name": "Test WER"
                                }
                            ]
                        },
                        {
                            "task": {
                                "type": "automatic-speech-recognition",
                                "name": "Automatic Speech Recognition"
                            },
                            "dataset": {
                                "name": "GigaSpeech",
                                "type": "speechcolab/gigaspeech",
                                "split": "test",
                                "args": {
                                    "language": "en"
                                }
                            },
                            "metrics": [
                                {
                                    "type": "wer",
                                    "value": 9.96,
                                    "name": "Test WER"
                                }
                            ]
                        },
                        {
                            "task": {
                                "type": "automatic-speech-recognition",
                                "name": "Automatic Speech Recognition"
                            },
                            "dataset": {
                                "name": "LibriSpeech (clean)",
                                "type": "librispeech_asr",
                                "config": "other",
                                "split": "test",
                                "args": {
                                    "language": "en"
                                }
                            },
                            "metrics": [
                                {
                                    "type": "wer",
                                    "value": 1.46,
                                    "name": "Test WER"
                                },
                                {
                                    "type": "wer",
                                    "value": 2.47,
                                    "name": "Test WER"
                                }
                            ]
                        },
                        {
                            "task": {
                                "type": "Automatic Speech Recognition",
                                "name": "automatic-speech-recognition"
                            },
                            "dataset": {
                                "name": "SPGI Speech",
                                "type": "kensho/spgispeech",
                                "config": "test",
                                "split": "test",
                                "args": {
                                    "language": "en"
                                }
                            },
                            "metrics": [
                                {
                                    "type": "wer",
                                    "value": 3.11,
                                    "name": "Test WER"
                                }
                            ]
                        },
                        {
                            "task": {
                                "type": "Automatic Speech Recognition",
                                "name": "automatic-speech-recognition"
                            },
                            "dataset": {
                                "name": "tedlium-v3",
                                "type": "LIUM/tedlium",
                                "config": "release1",
                                "split": "test",
                                "args": {
                                    "language": "en"
                                }
                            },
                            "metrics": [
                                {
                                    "type": "wer",
                                    "value": 3.92,
                                    "name": "Test WER"
                                }
                            ]
                        },
                        {
                            "task": {
                                "type": "automatic-speech-recognition",
                                "name": "Automatic Speech Recognition"
                            },
                            "dataset": {
                                "name": "Vox Populi",
                                "type": "facebook/voxpopuli",
                                "config": "en",
                                "split": "test",
                                "args": {
                                    "language": "en"
                                }
                            },
                            "metrics": [
                                {
                                    "type": "wer",
                                    "value": 5.39,
                                    "name": "Test WER"
                                }
                            ]
                        },
                        {
                            "task": {
                                "type": "Automatic Speech Recognition",
                                "name": "automatic-speech-recognition"
                            },
                            "dataset": {
                                "name": "Mozilla Common Voice 9.0",
                                "type": "mozilla-foundation/common_voice_9_0",
                                "config": "en",
                                "split": "test",
                                "args": {
                                    "language": "en"
                                }
                            },
                            "metrics": [
                                {
                                    "type": "wer",
                                    "value": 5.79,
                                    "name": "Test WER"
                                }
                            ]
                        }
                    ]
                }
            ]
        },
        "relevant_websites": [
            "https://img.shields.io/badge/Model_Arch-FastConformer--Transducer-lightgrey#model-badge)](#model-architecture",
            "https://img.shields.io/badge/Params-1.1B-lightgrey#model-badge)](#model-architecture",
            "https://img.shields.io/badge/Language-en-lightgrey#model-badge)](#datasets",
            "https://github.com/NVIDIA/NeMo",
            "https://www.suno.ai",
            "https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer",
            "https://github.com/NVIDIA/NeMo",
            "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav",
            "https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer",
            "https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py",
            "https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/fast-conformer_transducer_bpe.yaml",
            "https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py",
            "https://huggingface.co/spaces/hf-audio/open_asr_leaderboard",
            "https://developer.nvidia.com/riva",
            "https://huggingface.co/models?other=Riva",
            "https://developer.nvidia.com/riva#demos",
            "https://arxiv.org/abs/2305.05084",
            "https://github.com/google/sentencepiece",
            "https://github.com/NVIDIA/NeMo",
            "https://suno.ai",
            "https://huggingface.co/spaces/hf-audio/open_asr_leaderboard",
            "https://creativecommons.org/licenses/by/4.0",
            "https://creativecommons.org/licenses/by/4.0"
        ],
        "text": "Parakeet RNNT 1.1B (en)  [ | [ | [ parakeet-rnnt-1.1b is an ASR model that transcribes speech in lower case English alphabet. This model is jointly developed by NVIDIA NeMo and Suno.ai teams. It is an XXL version of FastConformer Transducer [1] (around 1.1B parameters) model. See the model architecture section and NeMo documentation for complete architecture details. NVIDIA NeMo: Training To train, fine-tune or play with the model you will need to install NVIDIA NeMo. We recommend you install it after youve installed latest PyTorch version. pip install nemo_toolkit[all]  How to Use this Model The model is available for use in the NeMo toolkit [3], and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset. Automatically instantiate the model Transcribing using Python First, lets get a sample wget Then simply do: asr_model.transcribe([2086-149220-0033.wav]) Transcribing many audio files shell python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py   pretrained_name=\\nvidia/parakeet-rnnt-1.1b\\   audio_dir=\\<DIRECTORY CONTAINING AUDIO FILES>\\ Input This model accepts 16000 Hz mono-channel audio (wav files) as input. Output This model provides transcribed speech as a string for a given audio sample. Model Architecture FastConformer [1] is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling. The model is trained in a multitask setup with a Transducer decoder (RNNT) loss. You may find more information on the details of FastConformer here: Fast-Conformer Model. Training The NeMo toolkit [3] was used for training the models for over several hundred epochs. These model are trained with this example script and this base config. The tokenizers for these models were built using the text transcripts of the train set with this script. Datasets The model was trained on 64K hours of English speech collected and prepared by NVIDIA NeMo and Suno teams. The training dataset consists of private subset with 40K hours of English speech plus 24K hours from the following public datasets:  Librispeech 960 hours of English speech Fisher Corpus Switchboard-1 Dataset WSJ-0 and WSJ-1 National Speech Corpus (Part 1, Part 6) VCTK VoxPopuli (EN) Europarl-ASR (EN) Multilingual Librispeech (MLS EN) - 2,000 hour subset Mozilla Common Voice (v7.0) Peoples Speech  - 12,000 hour subset  Performance The performance of Automatic Speech Recognition models is measuring using Word Error Rate. Since this dataset is trained on multiple domains and a much larger corpus, it will generally perform better at transcribing audio in general. The following tables summarizes the performance of the available models in this collection with the Transducer decoder. Performances of the ASR models are reported in terms of Word Error Rate (WER%) with greedy decoding.  |Version|Tokenizer|Vocabulary Size|AMI|Earnings-22|Giga Speech|LS test-clean|SPGI Speech|TEDLIUM-v3|Vox Populi|Common Voice| |---------|-----------------------|-----------------|---------------|---------------|------------|-----------|-----|-------|------|------| | 1.22.0  | SentencePiece Unigram | 1024 | 17.10 | 14.11 | 9.96 | 1.46 | 2.47 | 3.11 | 3.92 | 5.39 | 5.79 | These are greedy WER numbers without external LM. More details on evaluation can be found at HuggingFace ASR Leaderboard NVIDIA Riva: Deployment NVIDIA Riva, is an accelerated speech AI SDK deployable on-prem, in all clouds, multi-cloud, hybrid, on edge, and embedded.  Additionally, Riva provides:   World-class out-of-the-box accuracy for the most common languages with model checkpoints trained on proprietary data with hundreds of thousands of GPU-compute hours  Best in class accuracy with run-time word boosting (e.g., brand and product names) and customization of acoustic model, language model, and inverse text normalization  Streaming speech recognition, Kubernetes compatible scaling, and enterprise-grade support   Although this model isn\\u2019t supported yet by Riva, the list of supported models is here. Check out Riva live demo.  References [1] Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition [2] Google Sentencepiece Tokenizer [3] NVIDIA NeMo Toolkit [4] Suno.ai [5] HuggingFace ASR Leaderboard Licence License to use this model is covered by the CC-BY-4.0. By downloading the public and release version of the model, you accept the terms and conditions of the CC-BY-4.0 license.",
        "markdown_text": "\\n# Parakeet RNNT 1.1B (en)\\n\\n<style>\\nimg {\\n display: inline;\\n}\\n</style>\\n\\n[![Model architecture](https://img.shields.io/badge/Model_Arch-FastConformer--Transducer-lightgrey#model-badge)](#model-architecture)\\n| [![Model size](https://img.shields.io/badge/Params-1.1B-lightgrey#model-badge)](#model-architecture)\\n| [![Language](https://img.shields.io/badge/Language-en-lightgrey#model-badge)](#datasets)\\n\\n\\n`parakeet-rnnt-1.1b` is an ASR model that transcribes speech in lower case English alphabet. This model is jointly developed by [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) and [Suno.ai](https://www.suno.ai/) teams.\\nIt is an XXL version of FastConformer Transducer [1] (around 1.1B parameters) model.\\nSee the [model architecture](#model-architecture) section and [NeMo documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer) for complete architecture details.\\n\\n## NVIDIA NeMo: Training\\n\\nTo train, fine-tune or play with the model you will need to install [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). We recommend you install it after youve installed latest PyTorch version.\\n```\\npip install nemo_toolkit[all]\\n``` \\n\\n## How to Use this Model\\n\\nThe model is available for use in the NeMo toolkit [3], and can be used as a pre-trained checkpoint for inference or for fine-tuning on another dataset.\\n\\n### Automatically instantiate the model\\n\\n```python\\nimport nemo.collections.asr as nemo_asr\\nasr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=\\nvidia/parakeet-rnnt-1.1b\\)\\n```\\n\\n### Transcribing using Python\\nFirst, lets get a sample\\n```\\nwget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav\\n```\\nThen simply do:\\n```\\nasr_model.transcribe([2086-149220-0033.wav])\\n```\\n\\n### Transcribing many audio files\\n\\n```shell\\npython [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py \\n pretrained_name=\\nvidia/parakeet-rnnt-1.1b\\ \\n audio_dir=\\<DIRECTORY CONTAINING AUDIO FILES>\\\\n```\\n\\n### Input\\n\\nThis model accepts 16000 Hz mono-channel audio (wav files) as input.\\n\\n### Output\\n\\nThis model provides transcribed speech as a string for a given audio sample.\\n\\n## Model Architecture\\n\\nFastConformer [1] is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling. The model is trained in a multitask setup with a Transducer decoder (RNNT) loss. You may find more information on the details of FastConformer here: [Fast-Conformer Model](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/models.html#fast-conformer).\\n\\n## Training\\n\\nThe NeMo toolkit [3] was used for training the models for over several hundred epochs. These model are trained with this [example script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py) and this [base config](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/fast-conformer_transducer_bpe.yaml).\\n\\nThe tokenizers for these models were built using the text transcripts of the train set with this [script](https://github.com/NVIDIA/NeMo/blob/main/scripts/tokenizers/process_asr_text_tokenizer.py).\\n\\n### Datasets\\n\\nThe model was trained on 64K hours of English speech collected and prepared by NVIDIA NeMo and Suno teams.\\n\\nThe training dataset consists of private subset with 40K hours of English speech plus 24K hours from the following public datasets:\\n\\n- Librispeech 960 hours of English speech\\n- Fisher Corpus\\n- Switchboard-1 Dataset\\n- WSJ-0 and WSJ-1\\n- National Speech Corpus (Part 1, Part 6)\\n- VCTK\\n- VoxPopuli (EN)\\n- Europarl-ASR (EN)\\n- Multilingual Librispeech (MLS EN) - 2,000 hour subset\\n- Mozilla Common Voice (v7.0)\\n- Peoples Speech  - 12,000 hour subset\\n\\n## Performance\\n\\nThe performance of Automatic Speech Recognition models is measuring using Word Error Rate. Since this dataset is trained on multiple domains and a much larger corpus, it will generally perform better at transcribing audio in general.\\n\\nThe following tables summarizes the performance of the available models in this collection with the Transducer decoder. Performances of the ASR models are reported in terms of Word Error Rate (WER%) with greedy decoding. \\n\\n|**Version**|**Tokenizer**|**Vocabulary Size**|**AMI**|**Earnings-22**|**Giga Speech**|**LS test-clean**|**SPGI Speech**|**TEDLIUM-v3**|**Vox Populi**|**Common Voice**|\\n|---------|-----------------------|-----------------|---------------|---------------|------------|-----------|-----|-------|------|------|\\n| 1.22.0  | SentencePiece Unigram | 1024 | 17.10 | 14.11 | 9.96 | 1.46 | 2.47 | 3.11 | 3.92 | 5.39 | 5.79 |\\n\\nThese are greedy WER numbers without external LM. More details on evaluation can be found at [HuggingFace ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)\\n\\n## NVIDIA Riva: Deployment\\n\\n[NVIDIA Riva](https://developer.nvidia.com/riva), is an accelerated speech AI SDK deployable on-prem, in all clouds, multi-cloud, hybrid, on edge, and embedded. \\nAdditionally, Riva provides: \\n\\n* World-class out-of-the-box accuracy for the most common languages with model checkpoints trained on proprietary data with hundreds of thousands of GPU-compute hours \\n* Best in class accuracy with run-time word boosting (e.g., brand and product names) and customization of acoustic model, language model, and inverse text normalization \\n* Streaming speech recognition, Kubernetes compatible scaling, and enterprise-grade support \\n\\nAlthough this model isn\\u2019t supported yet by Riva, the [list of supported models is here](https://huggingface.co/models?other=Riva).  \\nCheck out [Riva live demo](https://developer.nvidia.com/riva#demos). \\n\\n## References\\n[1] [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084)\\n\\n[2] [Google Sentencepiece Tokenizer](https://github.com/google/sentencepiece)\\n\\n[3] [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)\\n\\n[4] [Suno.ai](https://suno.ai/)\\n\\n[5] [HuggingFace ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)\\n\\n\\n## Licence\\n\\nLicense to use this model is covered by the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). By downloading the public and release version of the model, you accept the terms and conditions of the [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) license.",
        "extraction_version": "v_2",
        "llm_extraction": {
            "model_name": "Parakeet RNNT 1.1B (en)",
            "model_framework": "NONE",
            "model_architecture": "RNN",
            "tasks": [
                "speech recognition"
            ],
            "training_strategy": "greedy",
            "parameters": "1.1B",
            "vocab_size": "NONE",
            "training_data": "proprietary data",
            "authors": [],
            "other": [
                "NVIDIA Riva",
                "world-class accuracy",
                "run-time word boosting",
                "customization",
                "streaming speech recognition",
                "Kubernetes compatible scaling",
                "enterprise-grade support",
                "Fast Conformer",
                "Google Sentencepiece Tokenizer",
                "NVIDIA NeMo Toolkit",
                "Suno.ai",
                "HuggingFace ASR Leaderboard",
                "CC-BY-4.0"
            ]
        },
        "papers_with_code": "successful",
        "performance": [
            {
                "id": "a2acf7e7-5480-4284-b7fa-dad0005bd24b",
                "best_rank": null,
                "metrics": {
                    "Word Error Rate (WER)": "5.8%"
                },
                "methodology": "parakeet-rnnt-1.1b",
                "uses_additional_data": true,
                "paper": "fast-conformer-with-linearly-scalable",
                "best_metric": null,
                "evaluated_on": "2023-05-08",
                "evaluation": "speech-recognition-on-common-voice-english",
                "benchmark_details": {
                    "id": "speech-recognition-on-common-voice-english",
                    "task": "speech-recognition",
                    "dataset": "common-voice-english",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "633910f8-dce8-4774-951c-75920e24f6dd",
                "best_rank": null,
                "metrics": {
                    "Word Error Rate (WER)": "1.46"
                },
                "methodology": "parakeet-rnnt-1.1b",
                "uses_additional_data": false,
                "paper": "fast-conformer-with-linearly-scalable",
                "best_metric": null,
                "evaluated_on": "2023-05-08",
                "evaluation": "speech-recognition-on-librispeech-test-clean",
                "benchmark_details": {
                    "id": "speech-recognition-on-librispeech-test-clean",
                    "task": "speech-recognition",
                    "dataset": "librispeech-test-clean",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "15c2410f-939a-4209-a859-110f8f051697",
                "best_rank": null,
                "metrics": {
                    "Word Error Rate (WER)": "2.48"
                },
                "methodology": "parakeet-rnnt-1.1b",
                "uses_additional_data": false,
                "paper": "fast-conformer-with-linearly-scalable",
                "best_metric": null,
                "evaluated_on": "2023-05-08",
                "evaluation": "speech-recognition-on-librispeech-test-other",
                "benchmark_details": {
                    "id": "speech-recognition-on-librispeech-test-other",
                    "task": "speech-recognition",
                    "dataset": "librispeech-test-other",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "6539fe3f-34ae-42d5-926e-5752a6ec9ad7",
                "best_rank": null,
                "metrics": {
                    "Word Error Rate (WER)": "3.11"
                },
                "methodology": "parakeet-rnnt-1.1b",
                "uses_additional_data": true,
                "paper": "fast-conformer-with-linearly-scalable",
                "best_metric": null,
                "evaluated_on": "2023-05-08",
                "evaluation": "speech-recognition-on-spgispeech",
                "benchmark_details": {
                    "id": "speech-recognition-on-spgispeech",
                    "task": "speech-recognition",
                    "dataset": "spgispeech",
                    "description": "",
                    "mirror_url": null
                }
            },
            {
                "id": "3bb81e2b-701c-4293-a507-83e53ca6fee8",
                "best_rank": null,
                "metrics": {
                    "Word Error Rate (WER)": "3.92"
                },
                "methodology": "parakeet-rnnt-1.1b",
                "uses_additional_data": true,
                "paper": "fast-conformer-with-linearly-scalable",
                "best_metric": null,
                "evaluated_on": "2023-05-08",
                "evaluation": "speech-recognition-on-tedlium",
                "benchmark_details": {
                    "id": "speech-recognition-on-tedlium",
                    "task": "speech-recognition",
                    "dataset": "tedlium",
                    "description": "",
                    "mirror_url": null
                }
            }
        ],
        "model_usage": {
            "llm_input": {
                "Parakeet RNNT 1.1B (en)": {
                    "Parakeet RNNT 1.1B (en)/ NVIDIA NeMo: Training": "To train, fine-tune or play with the model you will need to install NVIDIA NeMo. We recommend you install it after youve installed latest PyTorch version.\\npip install nemo_toolkit[all]",
                    "How to Use this Model/ Automatically instantiate the model": "",
                    "How to Use this Model/ Transcribing many audio files": "python [NEMO_GIT_FOLDER]/examples/asr/transcribe_speech.py \\n pretrained_name=\\nvidia/parakeet-rnnt-1.1b\\ \\n audio_dir=\\<DIRECTORY CONTAINING AUDIO FILES>\\,How to Use this Model/ Input: This model accepts 16000 Hz mono-channel audio (wav files) as input.",
                    "How to Use this Model/ Output": "This model provides transcribed speech as a string for a given audio sample.",
                    "Parakeet RNNT 1.1B (en)/ Model Architecture": "FastConformer [1] is an optimized version of the Conformer model with 8x depthwise-separable convolutional downsampling. The model is trained in a multitask setup with a Transducer decoder (RNNT) loss. You may find more information on the details of FastConformer here: Fast-Conformer Model."
                }
            },
            "usage": {
                "How to Use this Model/ Transcribing using Python": "First, lets get a sample\\nwget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav\\nThen simply do:\\nasr_model.transcribe([2086-149220-0033.wav])"
            },
            "model_function": [
                {
                    "code": "a",
                    "function_info": {
                        "return": null,
                        "function_name": "automatic_speech_recognition"
                    }
                }
            ]
        }
    }
};
