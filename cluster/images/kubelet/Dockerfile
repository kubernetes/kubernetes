FROM centos
ADD kubelet /kubelet
RUN chmod a+rx /kubelet
RUN cp /usr/bin/nsenter /nsenter

VOLUME /var/lib/docker
VOLUME /var/lib/kubelet
CMD [ "/kubelet" ]
