FROM ubuntu:14.04

RUN apt-get update -qy && apt-get install mutt offlineimap vim-nox abook elinks curl tmux cron zsh -y
ADD .muttrc /
ADD .offlineimaprc /
ADD .tmux.conf /
ADD mutt /.mutt
ADD vim /.vim
ADD vimrc /.vimrc
ADD crontab /etc/crontab
RUN chmod 644 /etc/crontab
RUN mkdir /Mail
RUN mkdir /.offlineimap
RUN echo "export TERM=screen-256color" >/.zshenv

CMD setsid cron; tmux -2
