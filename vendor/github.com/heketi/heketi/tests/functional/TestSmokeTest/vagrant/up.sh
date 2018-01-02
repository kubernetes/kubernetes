#!/bin/sh

vagrant up --no-provision $@
vagrant provision
