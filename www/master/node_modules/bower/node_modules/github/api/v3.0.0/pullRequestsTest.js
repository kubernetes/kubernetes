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

describe("[pullRequests]", function() {
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

    it("should successfully execute GET /repos/:user/:repo/pulls (getAll)",  function(next) {
        client.pullRequests.getAll(
            {
                user: "String",
                repo: "String",
                state: "String",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/pulls/:number (get)",  function(next) {
        client.pullRequests.get(
            {
                user: "String",
                repo: "String",
                number: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /repos/:user/:repo/pulls (create)",  function(next) {
        client.pullRequests.create(
            {
                user: "String",
                repo: "String",
                title: "String",
                body: "String",
                base: "String",
                head: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /repos/:user/:repo/pulls (createFromIssue)",  function(next) {
        client.pullRequests.createFromIssue(
            {
                user: "String",
                repo: "String",
                issue: "Number",
                base: "String",
                head: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute PATCH /repos/:user/:repo/pulls/:number (update)",  function(next) {
        client.pullRequests.update(
            {
                user: "String",
                repo: "String",
                number: "Number",
                state: "String",
                title: "String",
                body: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/pulls/:number/commits (getCommits)",  function(next) {
        client.pullRequests.getCommits(
            {
                user: "String",
                repo: "String",
                number: "Number",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/pulls/:number/files (getFiles)",  function(next) {
        client.pullRequests.getFiles(
            {
                user: "String",
                repo: "String",
                number: "Number",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/pulls/:number/merge (getMerged)",  function(next) {
        client.pullRequests.getMerged(
            {
                user: "String",
                repo: "String",
                number: "Number",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute PUT /repos/:user/:repo/pulls/:number/merge (merge)",  function(next) {
        client.pullRequests.merge(
            {
                user: "String",
                repo: "String",
                number: "Number",
                commit_message: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/pulls/:number/comments (getComments)",  function(next) {
        client.pullRequests.getComments(
            {
                user: "String",
                repo: "String",
                number: "Number",
                page: "Number",
                per_page: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/pulls/comments/:number (getComment)",  function(next) {
        client.pullRequests.getComment(
            {
                user: "String",
                repo: "String",
                number: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /repos/:user/:repo/pulls/:number/comments (createComment)",  function(next) {
        client.pullRequests.createComment(
            {
                user: "String",
                repo: "String",
                number: "Number",
                body: "String",
                commit_id: "String",
                path: "String",
                position: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /repos/:user/:repo/pulls/:number/comments (createCommentReply)",  function(next) {
        client.pullRequests.createCommentReply(
            {
                user: "String",
                repo: "String",
                number: "Number",
                body: "String",
                in_reply_to: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute PATCH /repos/:user/:repo/pulls/comments/:number (updateComment)",  function(next) {
        client.pullRequests.updateComment(
            {
                user: "String",
                repo: "String",
                number: "Number",
                body: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute DELETE /repos/:user/:repo/pulls/comments/:number (deleteComment)",  function(next) {
        client.pullRequests.deleteComment(
            {
                user: "String",
                repo: "String",
                number: "Number"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });
});
