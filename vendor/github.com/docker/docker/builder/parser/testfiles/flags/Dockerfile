FROM scratch
COPY foo /tmp/
COPY --user=me foo /tmp/
COPY --doit=true foo /tmp/
COPY --user=me --doit=true foo /tmp/
COPY --doit=true -- foo /tmp/
COPY -- foo /tmp/
CMD --doit [ "a", "b" ]
CMD --doit=true -- [ "a", "b" ]
CMD --doit -- [ ]
