FROM ubuntu:14.04

RUN apt-get update -qy && apt-get install tmux zsh weechat-curses -y

ADD .weechat /.weechat
ADD .tmux.conf /
RUN echo "export TERM=screen-256color" >/.zshenv

CMD zsh -c weechat
