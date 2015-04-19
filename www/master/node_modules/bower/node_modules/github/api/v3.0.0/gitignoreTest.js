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

describe("[gitignore]", function() {
    var client;

    beforeEach(function() {
        client = new Client({
            version: "3.0.0"
        });
    });

    it("should successfully execute GET /gitignore/templates (templates)",  function(next) {
        client.gitignore.templates(
            {},
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                Assert.ifError(err);
                Assert(Array.isArray(res));
                Assert(res.length > 10);
                next();
            }
        );
    });

    it("should successfully execute GET /gitignore/templates/:name (template)",  function(next) {
        client.gitignore.template(
            {
                name: "C"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                Assert.ifError(err);
                Assert('name' in res);
                Assert('source' in res);
                Assert(typeof res.source === 'string');
                next();
            }
        );
    });
});
