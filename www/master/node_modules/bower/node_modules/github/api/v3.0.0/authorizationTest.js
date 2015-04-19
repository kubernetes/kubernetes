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

describe("[authorization]", function() {
    var client;
    var token = "c286e38330e15246a640c2cf32a45ea45d93b2ba";

    this.timeout(10000);

    beforeEach(function() {
        client = new Client({
            version: "3.0.0"
        });
        client.authenticate({
            type: "basic",
            username: "mikedeboertest",
            password: "test1324"
        });
    });

    it("should successfully execute GET /authorizations (getAll)",  function(next) {
        client.authorization.create(
            {
                scopes: ["user", "public_repo", "repo", "repo:status", "delete_repo", "gist"],
                note: "Authorization created to unit tests auth",
                note_url: "https://github.com/ajaxorg/node-github"
            },
            function(err, res) {
                Assert.equal(err, null);
                var id = res.id;

                client.authorization.getAll(
                    {
                        page: "1",
                        per_page: "100"
                    },
                    function(err, res) {
                        Assert.equal(err, null);
                        Assert.equal(res.length, 1);

                        client.authorization["delete"](
                            {
                                id: id
                            },
                            function(err, res) {
                                Assert.equal(err, null);

                                client.authorization.getAll(
                                    {
                                        page: "1",
                                        per_page: "100"
                                    },
                                    function(err, res) {
                                        Assert.equal(err, null);
                                        Assert.equal(res.length, 0);

                                        next();
                                    }
                                );
                            }
                        );
                    }
                );
            }
        );
    });

    it("should successfully execute GET /authorizations/:id (get)",  function(next) {
        client.authorization.create(
            {
                scopes: ["user", "public_repo", "repo", "repo:status", "delete_repo", "gist"],
                note: "Authorization created to unit tests auth",
                note_url: "https://github.com/ajaxorg/node-github"
            },
            function(err, res) {
                Assert.equal(err, null);
                var id = res.id;

                client.authorization.get(
                    {
                        id: id
                    },
                    function(err, res) {
                        Assert.equal(err, null);
                        Assert.equal(res.id, id);
                        Assert.equal(res.note, "Authorization created to unit tests auth");
                        Assert.equal(res.note_url, "https://github.com/ajaxorg/node-github");

                        client.authorization["delete"](
                            {
                                id: id
                            },
                            function(err, res) {
                                Assert.equal(err, null);

                                client.authorization.get(
                                    {
                                        id: id
                                    },
                                    function(err, res) {
                                        Assert.equal(err.code, 404);
                                        next();
                                    }
                                );
                            }
                        );
                    }
                );
            }
        );
    });

    it("should successfully execute POST /authorizations (create)",  function(next) {
        client.authorization.create(
            {
                scopes: ["user", "public_repo", "repo", "repo:status", "delete_repo", "gist"],
                note: "Authorization created to unit tests auth",
                note_url: "https://github.com/ajaxorg/node-github"
            },
            function(err, res) {
                Assert.equal(err, null);
                var id = res.id;

                client.authorization.get(
                    {
                        id: id
                    },
                    function(err, res) {
                        Assert.equal(err, null);
                        Assert.equal(res.id, id);
                        Assert.equal(res.note, "Authorization created to unit tests auth");
                        Assert.equal(res.note_url, "https://github.com/ajaxorg/node-github");

                        client.authorization["delete"](
                            {
                                id: id
                            },
                            function(err, res) {
                                Assert.equal(err, null);

                                client.authorization.get(
                                    {
                                        id: id
                                    },
                                    function(err, res) {
                                        Assert.equal(err.code, 404);
                                        next();
                                    }
                                );
                            }
                        );
                    }
                );
            }
        );
    });

    it("should successfully execute PATCH /authorizations/:id (update)",  function(next) {
        client.authorization.create(
            {
                scopes: ["user", "public_repo", "repo", "repo:status", "delete_repo", "gist"],
                note: "Authorization created to unit tests auth",
                note_url: "https://github.com/ajaxorg/node-github"
            },
            function(err, res) {
                Assert.equal(err, null);
                var id = res.id;

                client.authorization.update(
                    {
                        id: id,
                        remove_scopes: ["repo"],
                        note: "changed"
                    },
                    function(err, res) {
                        Assert.equal(err, null);

                        client.authorization.get(
                            {
                                id: id
                            },
                            function(err, res) {
                                Assert.equal(err, null);
                                Assert.equal(res.id, id);
                                Assert.ok(res.scopes.indexOf("repo") === -1);
                                Assert.equal(res.note, "changed");
                                Assert.equal(res.note_url, "https://github.com/ajaxorg/node-github");

                                client.authorization["delete"](
                                    {
                                        id: id
                                    },
                                    function(err, res) {
                                        Assert.equal(err, null);

                                        client.authorization.get(
                                            {
                                                id: id
                                            },
                                            function(err, res) {
                                                Assert.equal(err.code, 404);
                                                next();
                                            }
                                        );
                                    }
                                );
                            }
                        );
                    }
                );
            }
        );
    });

    it("should successfully execute DELETE /authorizations/:id (delete)",  function(next) {
        client.authorization.create(
            {
                scopes: ["user", "public_repo", "repo", "repo:status", "delete_repo", "gist"],
                note: "Authorization created to unit tests auth",
                note_url: "https://github.com/ajaxorg/node-github"
            },
            function(err, res) {
                Assert.equal(err, null);
                var id = res.id;

                client.authorization["delete"](
                    {
                        id: id
                    },
                    function(err, res) {
                        Assert.equal(err, null);

                        client.authorization.get(
                            {
                                id: id
                            },
                            function(err, res) {
                                Assert.equal(err.code, 404);
                                next();
                            }
                        );
                    }
                );
            }
        );
    });
});
