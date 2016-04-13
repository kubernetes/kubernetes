FROM ubuntu:12.04

EXPOSE 27015
EXPOSE 27005
EXPOSE 26901
EXPOSE 27020

RUN apt-get update && apt-get install libc6-dev-i386 curl unzip -y
RUN mkdir -p /steam
RUN curl http://media.steampowered.com/client/steamcmd_linux.tar.gz | tar vxz -C /steam
ADD ./script /steam/script
RUN /steam/steamcmd.sh +runscript /steam/script
RUN curl http://mirror.pointysoftware.net/alliedmodders/mmsource-1.10.0-linux.tar.gz | tar vxz -C /steam/tf2/tf
RUN curl http://mirror.pointysoftware.net/alliedmodders/sourcemod-1.5.3-linux.tar.gz | tar vxz -C /steam/tf2/tf
ADD ./server.cfg /steam/tf2/tf/cfg/server.cfg
ADD ./ctf_2fort.cfg /steam/tf2/tf/cfg/ctf_2fort.cfg
ADD ./sourcemod.cfg /steam/tf2/tf/cfg/sourcemod/sourcemod.cfg
RUN rm -r /steam/tf2/tf/addons/sourcemod/configs
ADD ./configs /steam/tf2/tf/addons/sourcemod/configs
RUN mkdir -p /steam/tf2/tf/addons/sourcemod/translations/en
RUN cp /steam/tf2/tf/addons/sourcemod/translations/*.txt /steam/tf2/tf/addons/sourcemod/translations/en

CMD cd /steam/tf2 && ./srcds_run -port 27015 +ip 0.0.0.0 +map ctf_2fort -autoupdate -steam_dir /steam -steamcmd_script /steam/script +tf_bot_quota 12 +tf_bot_quota_mode fill
