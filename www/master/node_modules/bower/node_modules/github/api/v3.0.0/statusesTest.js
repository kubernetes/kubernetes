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

describe("[statuses]", function() {
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

    it("should successfully execute GET /repos/:user/:repo/commits/:sha/statuses (get)",  function(next) {
        client.statuses.get(
            {
                user: "mikedeboer",
                repo: "node-github",
                sha: "30d607d8fd8002427b61273f25d442c233cbf631"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/commits/:sha/status (get)",  function(next) {
        client.statuses.getCombined(
            {
                user: "mikedeboer",
                repo: "node-github",
                sha: "30d607d8fd8002427b61273f25d442c233cbf631"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /repos/:user/:repo/statuses/:sha (create)",  function(next) {
        client.statuses.create(
            {
                user: "String",
                repo: "String",
                sha: "String",
                state: "String",
                target_url: "String",
                description: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });
});
