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

describe("[misc]", function() {
    var client;

    beforeEach(function() {
        client = new Client({
            version: "3.0.0"
        });
    });

    it("should successfully execute GET /emojis (emojis)",  function(next) {
        client.misc.emojis(
            {},
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                Assert.ifError(err);
                // A common emoji on github
                Assert('shipit' in res);
                next();
            }
        );
    });

    it("should successfully execute GET /meta (meta)",  function(next) {
        client.misc.meta(
            {},
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                Assert('hooks' in res);
                Assert('git' in res);
                next();
            }
        );
    });

    it("should successfully execute GET /rate_limit (rateLimit)",  function(next) {
        client.misc.rateLimit(
            {},
            function(err, res) {
                Assert.equal(err, null);
                Assert('resources' in res);
                Assert('core' in res.resources);
                Assert(typeof res.resources.core.limit === 'number');
                Assert(typeof res.resources.core.remaining === 'number');
                Assert(typeof res.resources.core.reset === 'number');
                Assert('search' in res.resources);
                Assert(typeof res.resources.search.limit === 'number');
                Assert(typeof res.resources.search.remaining === 'number');
                Assert(typeof res.resources.search.reset === 'number');
                next();
            }
        );
    });
});
