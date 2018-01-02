#!/usr/bin/env python

from ct.client import log_client
from twisted.internet import reactor
from twisted.web import client as twisted_client


def sth_callback(sth):
    print sth


def stop_callback(ignored):
    reactor.stop()


def get_sth():
    agent = twisted_client.Agent(reactor)
    client = log_client.AsyncLogClient(agent, "https://ct.googleapis.com/pilot")
    d = client.get_sth()
    # Print the STH on success.
    d.addCallback(sth_callback)
    # Stop the reactor whether we succeeded or not.
    d.addBoth(stop_callback)


if __name__ == "__main__":
    # Schedule the call.
    get_sth()
    # Start the event loop.
    reactor.run()
