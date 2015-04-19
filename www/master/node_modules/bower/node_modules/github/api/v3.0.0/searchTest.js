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

describe("[search]", function() {
    var client;
    var token = "c286e38330e15246a640c2cf32a45ea45d93b2ba";

    beforeEach(function() {
        client = new Client({
            version: "3.0.0"
        });
        /*client.authenticate({
            type: "oauth",
            token: token
        });*/
    });

    it("should successfully execute GET /search/issues/:q (issues)",  function(next) {
        client.search.issues(
            {
                q: ['macaroni', 'repo:mikedeboertest/node_chat', 'state:open'].join('+')
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.equal(res.items.length, 1);
                var issue = res.items[0];
                Assert.equal(issue.title, "My First Issue");
                Assert.equal(issue.state, "open");

                next();
            }
        );
    });

    it("should successfully execute GET /search/repositories/:q (repos)",  function(next) {
        client.search.repos(
            {
                q: ['pasta', 'language:JavaScript'].join('+')
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.ok(res.items.length > 0);
                Assert.equal(res.items[0].language, "JavaScript");

                next();
            }
        );
    });

    it("should successfully execute GET /search/users/:q (users)",  function(next) {
        client.search.users(
            {
                q: "mikedeboer"
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.equal(res.items.length, 2);
                var user = res.items[0];
                Assert.equal(user.login, "mikedeboer");

                client.search.users(
                    {
                        q: "location:Jyväskylä"
                    },
                    function(err, res) {
                        Assert.equal(err, null);
                        //XXX: this is likely to change often. I added this for
                        //     issue #159.
                        Assert.equal(res.items.length, 30);
                        var user = res.items[0];
                        Assert.equal(user.login, "bebraw");

                        next();
                    }
                );
            }
        );
    });

    /*it("should successfully execute GET /search/user/email/:email (email)",  function(next) {
        client.search.email(
            {
                email: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });*/
});
