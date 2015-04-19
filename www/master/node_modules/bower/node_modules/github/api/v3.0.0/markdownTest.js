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

describe("[markdown]", function() {
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

    it("should successfully execute POST /markdown (render)",  function(next) {
        client.markdown.render(
            {
                text: "Hello world github/linguist#1 **cool**, and #1!",
                mode: "gfm",
                context: "github/gollem"
            },
            function(err, res) {
                Assert.equal(err, null);
                console.log(res);
                next();
            }
        );
    });
});
