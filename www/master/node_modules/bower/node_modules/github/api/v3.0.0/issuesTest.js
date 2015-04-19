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

describe("[issues]", function() {
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

    it("should successfully execute GET /issues (getAll)",  function(next) {
        client.issues.getAll(
            {
                filter: "created",
                state: "open",
                labels: "",
                sort: "updated",
                direction: "asc"
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.equal(res.length, 1);
                var issue = res[0];
                Assert.equal(issue.title, "My First Issue");
                Assert.equal(issue.number, 2);
                Assert.equal(issue.state, "open");
                Assert.equal(issue.body, "Willing to start a debate on the best recipe of macaroni.");
                Assert.equal(issue.assignee.login, "mikedeboertest");

                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/issues (repoIssues)",  function(next) {
        client.issues.repoIssues(
            {
                user: "mikedeboertest",
                repo: "node_chat",
                state: "open",
                sort: "updated",
                direction: "asc"
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.equal(res.length, 1);
                var issue = res[0];
                Assert.equal(issue.title, "My First Issue");
                Assert.equal(issue.number, 2);
                Assert.equal(issue.state, "open");
                Assert.equal(issue.body, "Willing to start a debate on the best recipe of macaroni.");
                Assert.equal(issue.assignee.login, "mikedeboertest");

                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/issues/:number (getRepoIssue)",  function(next) {
        client.issues.getRepoIssue(
            {
                user: "mikedeboertest",
                repo: "node_chat",
                number: 2
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.equal(res.title, "My First Issue");
                Assert.equal(res.number, 2);
                Assert.equal(res.state, "open");
                Assert.equal(res.body, "Willing to start a debate on the best recipe of macaroni.");
                Assert.equal(res.assignee.login, "mikedeboertest");

                next();
            }
        );
    });
/*
    it("should successfully execute POST /repos/:user/:repo/issues (create)",  function(next) {
        client.issues.create(
            {
                user: "String",
                repo: "String",
                title: "String",
                body: "String",
                assignee: "String",
                milestone: "Number",
                labels: "Json"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute PATCH /repos/:user/:repo/issues/:number (edit)",  function(next) {
        client.issues.edit(
            {
                user: "String",
                repo: "String",
                number: "Number",
                title: "String",
                body: "String",
                assignee: "String",
                milestone: "Number",
                labels: "Json"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/issues/:number/comments (getComments)",  function(next) {
        client.issues.getComments(
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

    it("should successfully execute GET /repos/:user/:repo/issues/comments/:id (getComment)",  function(next) {
        client.issues.getComment(
            {
                user: "String",
                repo: "String",
                id: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /repos/:user/:repo/issues/:number/comments (createComment)",  function(next) {
        client.issues.createComment(
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

    it("should successfully execute PATCH /repos/:user/:repo/issues/comments/:id (editComment)",  function(next) {
        client.issues.editComment(
            {
                user: "String",
                repo: "String",
                id: "String",
                body: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute DELETE /repos/:user/:repo/issues/comments/:id (deleteComment)",  function(next) {
        client.issues.deleteComment(
            {
                user: "String",
                repo: "String",
                id: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/issues/:number/events (getEvents)",  function(next) {
        client.issues.getEvents(
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

    it("should successfully execute GET /repos/:user/:repo/issues/events (getRepoEvents)",  function(next) {
        client.issues.getRepoEvents(
            {
                user: "String",
                repo: "String",
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

    it("should successfully execute GET /repos/:user/:repo/issues/events/:id (getEvent)",  function(next) {
        client.issues.getEvent(
            {
                user: "String",
                repo: "String",
                id: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/labels (getLabels)",  function(next) {
        client.issues.getLabels(
            {
                user: "String",
                repo: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/labels/:name (getLabel)",  function(next) {
        client.issues.getLabel(
            {
                user: "String",
                repo: "String",
                name: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /repos/:user/:repo/labels (createLabel)",  function(next) {
        client.issues.createLabel(
            {
                user: "String",
                repo: "String",
                name: "String",
                color: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute POST /repos/:user/:repo/labels/:name (updateLabel)",  function(next) {
        client.issues.updateLabel(
            {
                user: "String",
                repo: "String",
                name: "String",
                color: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute DELETE /repos/:user/:repo/labels/:name (deleteLabel)",  function(next) {
        client.issues.deleteLabel(
            {
                user: "String",
                repo: "String",
                name: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:user/:repo/milestones (getAllMilestones)",  function(next) {
        client.issues.getAllMilestones(
            {
                user: "String",
                repo: "String",
                state: "String",
                sort: "String",
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

    it("should successfully execute GET /repos/:user/:repo/milestones/:number (getMilestone)",  function(next) {
        client.issues.getMilestone(
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

    it("should successfully execute POST /repos/:user/:repo/milestones (createMilestone)",  function(next) {
        client.issues.createMilestone(
            {
                user: "String",
                repo: "String",
                title: "String",
                state: "String",
                description: "String",
                due_on: "Date"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute PATCH /repos/:user/:repo/milestones/:number (updateMilestone)",  function(next) {
        client.issues.updateMilestone(
            {
                user: "String",
                repo: "String",
                number: "Number",
                title: "String",
                state: "String",
                description: "String",
                due_on: "Date"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute DELETE /repos/:user/:repo/milestones/:number (deleteMilestone)",  function(next) {
        client.issues.deleteMilestone(
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
    });*/
});
