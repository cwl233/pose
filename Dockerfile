FROM openpose:latest
MAINTAINER lly <liuliuy@mail.ustc.edu.cn>

WORKDIR /openpose/build/examples/tutorial_api_python
COPY openpose_lly .
ENTRYPOINT ["python3","linke_vision_openpose.py"]
