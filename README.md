# Prepare for the venv environment
Create and activate the venv environment.
```bash
python -m venv venv
. venv/bin/activate
pip install --upgrade pip
```

Use the following command to install KIE dependencies.
```bash
pip install -r requirements.txt
pip install -r ppstructure/kie/requirements.txt
pip install paddleocr -U
```

Install PaddlePaddle.
```bash
# If you have CUDA 9 or CUDA 10 installed on your machine, please run the following command to install
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
# If you have no available GPU on your machine, please run the following command to install the CPU version
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

## Prepare for the dataset
```bash
mkdir train_data
cd train_data
# download and uncompress the dataset
wget https://paddleocr.bj.bcebos.com/ppstructure/dataset/XFUND.tar && tar -xf XFUND.tar
cd ..
```
- `train_data` directory is then created.

## Download models for `Inference using PaddleInference`
Firstly, download the inference SER inference model.
```bash
mkdir inference
cd inference
wget https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/ser_vi_layoutxlm_xfund_infer.tar && tar -xf ser_vi_layoutxlm_xfund_infer.tar
wget https://paddleocr.bj.bcebos.com/ppstructure/models/vi_layoutxlm/re_vi_layoutxlm_xfund_infer.tar && tar -xf re_vi_layoutxlm_xfund_infer.tar
cd ..
```
- `inference` directory is then created.

## SER inference
```bash
cd ppstructure
python3 kie/predict_kie_token_ser.py \
  --kie_algorithm=LayoutXLM \
  --ser_model_dir=../inference/ser_vi_layoutxlm_xfund_infer \
  --image_dir=./docs/kie/input/zh_val_42.jpg \
  --ser_dict_path=../train_data/XFUND/class_list_xfun.txt \
  --vis_font_path=../doc/fonts/simfang.ttf \
  --ocr_order_method="tb-yx"
```
- The visual results and text file will be saved in directory `ppstructure/output`.

## RE inference
```bash
cd ppstructure
python3 kie/predict_kie_token_ser_re.py \
  --kie_algorithm=LayoutXLM \
  --re_model_dir=../inference/re_vi_layoutxlm_xfund_infer \
  --ser_model_dir=../inference/ser_vi_layoutxlm_xfund_infer \
  --use_visual_backbone=False \
  --image_dir=./docs/kie/input/zh_val_42.jpg \
  --ser_dict_path=../train_data/XFUND/class_list_xfun.txt \
  --vis_font_path=../doc/fonts/simfang.ttf \
  --ocr_order_method="tb-yx"
```
- The visual results and text file will be saved in directory `ppstructure/output`.

If you want to use a custom ocr model, you can set it through the following fields.
* --det_model_dir: the detection inference model path
* --rec_model_dir: the recognition inference model path

# API development
Install dependencies.
```bash
pip install fastapi
pip install python-multipart
```