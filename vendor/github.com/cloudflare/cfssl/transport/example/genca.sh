#!/bin/sh

cfssl gencert -initca ca.json | cfssljson -bare ca

