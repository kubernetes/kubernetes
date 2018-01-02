FROM cpuguy83/ubuntu
ENV NAGIOS_HOME /opt/nagios
ENV NAGIOS_USER nagios
ENV NAGIOS_GROUP nagios
ENV NAGIOS_CMDUSER nagios
ENV NAGIOS_CMDGROUP nagios
ENV NAGIOSADMIN_USER nagiosadmin
ENV NAGIOSADMIN_PASS nagios
ENV APACHE_RUN_USER nagios
ENV APACHE_RUN_GROUP nagios
ENV NAGIOS_TIMEZONE UTC

RUN sed -i 's/universe/universe multiverse/' /etc/apt/sources.list
RUN apt-get update && apt-get install -y iputils-ping netcat build-essential snmp snmpd snmp-mibs-downloader php5-cli apache2 libapache2-mod-php5 runit bc postfix bsd-mailx
RUN ( egrep -i  "^${NAGIOS_GROUP}" /etc/group || groupadd $NAGIOS_GROUP ) && ( egrep -i "^${NAGIOS_CMDGROUP}" /etc/group || groupadd $NAGIOS_CMDGROUP )
RUN ( id -u $NAGIOS_USER || useradd --system $NAGIOS_USER -g $NAGIOS_GROUP -d $NAGIOS_HOME ) && ( id -u $NAGIOS_CMDUSER || useradd --system -d $NAGIOS_HOME -g $NAGIOS_CMDGROUP $NAGIOS_CMDUSER )

ADD http://downloads.sourceforge.net/project/nagios/nagios-3.x/nagios-3.5.1/nagios-3.5.1.tar.gz?r=http%3A%2F%2Fwww.nagios.org%2Fdownload%2Fcore%2Fthanks%2F%3Ft%3D1398863696&ts=1398863718&use_mirror=superb-dca3 /tmp/nagios.tar.gz
RUN cd /tmp && tar -zxvf nagios.tar.gz && cd nagios  && ./configure --prefix=${NAGIOS_HOME} --exec-prefix=${NAGIOS_HOME} --enable-event-broker --with-nagios-command-user=${NAGIOS_CMDUSER} --with-command-group=${NAGIOS_CMDGROUP} --with-nagios-user=${NAGIOS_USER} --with-nagios-group=${NAGIOS_GROUP} && make all && make install && make install-config && make install-commandmode && cp sample-config/httpd.conf /etc/apache2/conf.d/nagios.conf
ADD http://www.nagios-plugins.org/download/nagios-plugins-1.5.tar.gz /tmp/
RUN cd /tmp && tar -zxvf nagios-plugins-1.5.tar.gz && cd nagios-plugins-1.5 && ./configure --prefix=${NAGIOS_HOME} && make && make install

RUN sed -i.bak 's/.*\=www\-data//g' /etc/apache2/envvars
RUN export DOC_ROOT="DocumentRoot $(echo $NAGIOS_HOME/share)"; sed -i "s,DocumentRoot.*,$DOC_ROOT," /etc/apache2/sites-enabled/000-default

RUN ln -s ${NAGIOS_HOME}/bin/nagios /usr/local/bin/nagios && mkdir -p /usr/share/snmp/mibs && chmod 0755 /usr/share/snmp/mibs && touch /usr/share/snmp/mibs/.foo

RUN echo "use_timezone=$NAGIOS_TIMEZONE" >> ${NAGIOS_HOME}/etc/nagios.cfg && echo "SetEnv TZ \"${NAGIOS_TIMEZONE}\"" >> /etc/apache2/conf.d/nagios.conf

RUN mkdir -p ${NAGIOS_HOME}/etc/conf.d && mkdir -p ${NAGIOS_HOME}/etc/monitor && ln -s /usr/share/snmp/mibs ${NAGIOS_HOME}/libexec/mibs
RUN echo "cfg_dir=${NAGIOS_HOME}/etc/conf.d" >> ${NAGIOS_HOME}/etc/nagios.cfg
RUN echo "cfg_dir=${NAGIOS_HOME}/etc/monitor" >> ${NAGIOS_HOME}/etc/nagios.cfg
RUN download-mibs && echo "mibs +ALL" > /etc/snmp/snmp.conf

RUN sed -i 's,/bin/mail,/usr/bin/mail,' /opt/nagios/etc/objects/commands.cfg && \
  sed -i 's,/usr/usr,/usr,' /opt/nagios/etc/objects/commands.cfg
RUN cp /etc/services /var/spool/postfix/etc/

RUN mkdir -p /etc/sv/nagios && mkdir -p /etc/sv/apache && rm -rf /etc/sv/getty-5 && mkdir -p /etc/sv/postfix
ADD nagios.init /etc/sv/nagios/run
ADD apache.init /etc/sv/apache/run
ADD postfix.init /etc/sv/postfix/run
ADD postfix.stop /etc/sv/postfix/finish

ADD start.sh /usr/local/bin/start_nagios

ENV APACHE_LOCK_DIR /var/run
ENV APACHE_LOG_DIR /var/log/apache2

EXPOSE 80

VOLUME ["/opt/nagios/var", "/opt/nagios/etc", "/opt/nagios/libexec", "/var/log/apache2", "/usr/share/snmp/mibs"]

CMD ["/usr/local/bin/start_nagios"]
