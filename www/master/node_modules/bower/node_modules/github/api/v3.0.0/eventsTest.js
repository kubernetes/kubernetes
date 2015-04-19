/*
 * Copyright 2012 Cloud9 IDE, Inc.
 *
 * This product includes software developed by
 * Cloud9 IDE, Inc (http://c9.io).
 *
 * Author: Mike de Boer <info@mikedeboer.nl>
 */

"use strict";

var Assert = require("assert");
var Client = require("./../../index");

describe("[events]", function() {
    var client;
    var token = "c286e38330e15246a640c2cf32a45ea45d93b2ba";

    beforeEach(function() {
        client = new Client({
            version: "3.0.0"
        });
        client.authenticate({
            type: "oauth",
            token: token
        });
    });

    it("should successfully execute GET /events (get)",  function(next) {
        client.events.get(
            {
                page: 1,
                per_page: 30
            },
            function(err, res) {
                // other assertions go here
                Assert.equal(err, null);
                Assert.ok(res.length > 1);
                Assert.equal(typeof res[0].type, "string");
                Assert.equal(typeof res[0].created_at, "string");
                Assert.equal(typeof res[0]["public"], "boolean");
                Assert.equal(typeof res[0].id, "string");
                Assert.ok("actor" in res[0]);
                Assert.ok("repo" in res[0]);
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/events (getFromRepo)",  function(next) {
        client.events.getFromRepo(
            {
                user: "mikedeboertest",
                repo: "node_chat"
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.equal(res.length, 5);
                // this is the lastly listed event
                var last = res.pop();
                Assert.equal(last.type, "ForkEvent");
                Assert.equal(last.created_at, "2012-10-05T15:03:11Z");
                Assert.equal(last.id, "1607304921");
                Assert.equal(last["public"], true);
                Assert.equal(last.actor.login, "mikedeboer");
                Assert.equal(last.repo.name, "mikedeboertest/node_chat");
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/issues/events (getFromRepoIssues)",  function(next) {
        client.events.getFromRepoIssues(
            {
                user: "mikedeboertest",
                repo: "node_chat"
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.equal(res.length, 4);
                // this is the lastly listed event
                var last = res.pop();
                Assert.equal(last.event, "referenced");
                Assert.equal(last.created_at, "2012-10-05T15:05:31Z");
                Assert.equal(last.id, "26276344");
                Assert.equal(last.actor.login, "mikedeboertest");
                Assert.equal(last.issue.title, "Macaroni");
                Assert.equal(last.issue.number, 1);
                Assert.equal(last.issue.state, "closed");
                next();
            }
        );
    });

    it("should successfully execute GET /networks/:user/:repo/events (getFromRepoNetwork)",  function(next) {
        client.events.getFromRepoNetwork(
            {
                user: "mikedeboertest",
                repo: "node_chat"
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.ok(res.length > 1);
                var last = res.pop();
                Assert.equal(typeof last.id, "string");
                Assert.equal(typeof last.created_at, "string");
                Assert.equal(typeof last.actor, "object");
                next();
            }
        );
    });

    it("should successfully execute GET /orgs/:org/events (getFromOrg)",  function(next) {
        client.events.getFromOrg(
            {
                org: "ajaxorg"
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.ok(res.length > 1);
                var last = res.pop();
                Assert.equal(typeof last.id, "string");
                Assert.equal(typeof last.created_at, "string");
                Assert.equal(typeof last.actor, "object");
                next();
            }
        );
    });

    it("should successfully execute GET /users/:user/received_events (getReceived)",  function(next) {
        client.events.getReceived(
            {
                user: "mikedeboertest"
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.ok(res.length > 0);
                var last = res.pop();
                Assert.equal(typeof last.id, "string");
                Assert.equal(typeof last.created_at, "string");
                Assert.equal(typeof last.actor, "object");
                next();
            }
        );
    });

    it("should successfully execute GET /users/:user/received_events/public (getReceivedPublic)",  function(next) {
        client.events.getReceivedPublic(
            {
                user: "mikedeboertest"
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.ok(res.length > 0);
                var last = res.pop();
                Assert.equal(typeof last.id, "string");
                Assert.equal(typeof last.created_at, "string");
                Assert.equal(typeof last.actor, "object");
                next();
            }
        );
    });

    it("should successfully execute GET /users/:user/events (getFromUser)",  function(next) {
        client.events.getFromUser(
            {
                user: "mikedeboertest"
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.ok(res.length > 1);
                var last = res.pop();
                Assert.equal(typeof last.id, "string");
                Assert.equal(typeof last.created_at, "string");
                Assert.equal(typeof last.actor, "object");
                next();
            }
        );
    });

    it("should successfully execute GET /users/:user/events/public (getFromUserPublic)",  function(next) {
        client.events.getFromUserPublic(
            {
                user: "mikedeboertest"
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.ok(res.length > 1);
                var last = res.pop();
                Assert.equal(typeof last.id, "string");
                Assert.equal(typeof last.created_at, "string");
                Assert.equal(typeof last.actor, "object");
                next();
            }
        );
    });

    it("should successfully execute GET /users/:user/events/orgs/:org (getFromUserOrg)",  function(next) {
        client.events.getFromUserOrg(
            {
                user: "mikedeboer",
                org: "ajaxorg"
            },
            function(err, res) {
                // we're not logged in as `mikedeboer` right now, so github API does not allow
                // us to see the resource.
                Assert.equal(err.code, 404);
                next();
            }
        );
    });
});
