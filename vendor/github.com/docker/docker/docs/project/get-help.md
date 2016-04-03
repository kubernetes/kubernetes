<!--[metadata]>
+++
title = "Where to chat or get help"
description = "Describes Docker's communication channels"
keywords = ["IRC, Google group, Twitter, blog,  Stackoverflow"]
[menu.main]
parent = "mn_opensource"
+++
<![end-metadata]-->

<style type="text/css">
/* @TODO add 'no-zebra' table-style to the docs-base stylesheet */
/* Table without "zebra" striping */
.content-body table.no-zebra tr {
  background-color: transparent;
}
</style>

# Where to chat or get help

There are several communications channels you can use to chat with Docker
community members and developers.

<table>
  <col width="25%">
  <col width="75%">
  <tr>
    <td>Internet Relay Chat (IRC)</th>
    <td>
      <p>
        IRC a direct line to our most knowledgeable Docker users.
        The <code>#docker</code> and <code>#docker-dev</code> group on 
        <strong>irc.freenode.net</strong>. IRC was first created in 1988. 
        So, it is a rich chat protocol but it can overwhelm new users. You can search
        <a href="https://botbot.me/freenode/docker/#" target="_blank">our chat archives</a>.
      </p>
      Read our IRC quickstart guide below for an easy way to get started.
    </td>
  </tr>
  <tr>
    <td>Google Groups</td>
    <td>
      There are two groups.
      <a href="https://groups.google.com/forum/#!forum/docker-user" target="_blank">Docker-user</a>
      is for people using Docker containers. 
      The <a href="https://groups.google.com/forum/#!forum/docker-dev" target="_blank">docker-dev</a> 
      group is for contributors and other people contributing to the Docker 
      project.
    </td>
  </tr>
  <tr>
    <td>Twitter</td>
    <td>
      You can follow <a href="https://twitter.com/docker/" target="_blank">Docker's twitter</a>
      to get updates on our products. You can also tweet us questions or just 
      share blogs or stories.
    </td>
  </tr>
  <tr>
    <td>Stack Overflow</td>
    <td>
      Stack Overflow has over 7000K Docker questions listed. We regularly 
      monitor <a href="http://stackoverflow.com/search?tab=newest&q=docker" target="_blank">Docker questions</a>
      and so do many other knowledgeable Docker users.
    </td>
  </tr>
</table>


## IRC Quickstart

IRC can also be overwhelming for new users. This quickstart shows you 
the easiest way to connect to IRC. 

1. In your browser open <a href="http://webchat.freenode.net" target="_blank">http://webchat.freenode.net</a>

    ![Login screen](/project/images/irc_connect.png)


2. Fill out the form.

    <table class="no-zebra" style="width: auto">
      <tr>
        <td><b>Nickname</b></td>
        <td>The short name you want to be known as in IRC.</td>
      </tr>
      <tr>
        <td><b>Channels</b></td>
        <td><code>#docker</code></td>
      </tr>
      <tr>
        <td><b>reCAPTCHA</b></td>
        <td>Use the value provided.</td>
      </tr>
    </table>

3. Click "Connect".

    The system connects you to chat. You'll see a lot of text. At the bottom of
    the display is a command line. Just above the command line the system asks 
    you to register.

    ![Login screen](/project/images/irc_after_login.png)


4. In the command line, register your nickname.

        /msg NickServ REGISTER password youremail@example.com

    ![Login screen](/project/images/register_nic.png)

    The IRC system sends an email to the address you
    enter. The email contains instructions for completing your registration.

5. Open your mail client and look for the email.

    ![Login screen](/project/images/register_email.png)

6. Back in the browser, complete the registration according to the email.

        /msg NickServ VERIFY REGISTER moxiegirl_ acljtppywjnr

7. Join the `#docker` group using the following command.

        /j #docker

    You can also join the `#docker-dev` group.

        /j #docker-dev

8. To ask questions to the channel just type messages in the command line.

	![Login screen](/project/images/irc_chat.png)

9. To quit, close the browser window.


### Tips and learning more about IRC

Next time you return to log into chat, you'll need to re-enter your password 
on the command line using this command:

    /msg NickServ identify <password>

If you forget or lose your password see <a
href="https://freenode.net/faq.shtml#sendpass" target="_blank">the FAQ on
freenode.net</a> to learn how to recover it.

This quickstart was meant to get you up and into IRC very quickly. If you find 
IRC useful there is a lot more to learn. Drupal, another open source project, 
actually has <a href="https://www.drupal.org/irc/setting-up" target="_blank">
written a lot of good documentation about using IRC</a> for their project 
(thanks Drupal!).
