/*
 * Copyright 2012 Cloud9 IDE, Inc.
 *
 * This product includes software developed by
 * Cloud9 IDE, Inc (http://c9.io).
 *
 * Author: Mike de Boer <mike@c9.io>
 */

"use strict";

var Assert = require("assert");
var Client = require("./../index");

describe("[client]", function() {
    var client;
    var token = "e5a4a27487c26e571892846366de023349321a73";

    beforeEach(function() {
        client = new Client({
            version: "3.0.0"
        });
        /*client.authenticate({
            type: "oauth",
            token: token
        });*/
    });

    it("should successfully execute GET /authorizations (getAll)",  function(next) {
        // `aseemk` has two pages of followers right now.
        client.user.getFollowers(
            {
                user: "aseemk"
            },
            function(err, res) {
                Assert.equal(err, null);

                Assert.ok(!!client.hasNextPage(res));
                Assert.ok(!!client.hasLastPage(res));
                Assert.ok(!client.hasPreviousPage(res));

                client.getNextPage(res, function(err, res) {
                    Assert.equal(err, null);

                    Assert.ok(!!client.hasPreviousPage(res));
                    Assert.ok(!!client.hasFirstPage(res));
                    Assert.ok(!client.hasNextPage(res));
                    Assert.ok(!client.hasLastPage(res));

                    client.getPreviousPage(res.meta.link, function(err, res) {
                        Assert.equal(err, null);

                        Assert.ok(!!client.hasNextPage(res));
                        Assert.ok(!!client.hasLastPage(res));
                        Assert.ok(!client.hasPreviousPage(res));
                        next();
                    });
                });
            }
        );
    });
});
