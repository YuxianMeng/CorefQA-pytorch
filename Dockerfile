ARG base_image="gcr.io/tpu-pytorch/xla:r1.5"
FROM "${base_image}"


RUN apt-get update --fix-missing
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8


COPY scripts/tpu_bert_base.sh tpu_bert_base.sh
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


ENTRYPOINT ["bash", "tpu_bert_base.sh"]
CMD ["bash"]