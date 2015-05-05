FROM centos
ADD kubelet /kubelet
RUN chmod a+rx /kubelet
ADD nsenter /nsenter
RUN chmod a+rx /nsenter

VOLUME /var/lib/docker
VOLUME /var/lib/kubelet
CMD [ "/kubelet" ]
