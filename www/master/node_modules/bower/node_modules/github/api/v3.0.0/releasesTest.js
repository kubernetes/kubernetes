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

describe("[releases]", function() {
    var client;
    var token = "c286e38330e15246a640c2cf32a45ea45d93b2ba";

    var owner = "greggman";
    var repo  = "test";
    var haveWriteAccess = true;       // set to false if the authenticated person below does not have write access to the repo above
    var releaseIdWithAsset = 393621;  // Some release id from the repo above that has at least 1 asset.

    var releaseId;      // release id found when listing releases. Used for get release
    var newReleaseId;   // release id created when creating release, used for edit and delete release
    var assetId;        // asset id found when listing assets. Used for get asset
    var newAssetId;     // asset id used when creating asset. Used for edit and delete asset

    beforeEach(function() {
        client = new Client({
            version: "3.0.0"
        });
        client.authenticate({
            type: "oauth",
            token: token
        });
    });

    it("should successfully execute GET /repos/:owner/:repo/releases (listReleases)",  function(next) {
        client.releases.listReleases(
            {
              owner: owner,
              repo: repo,
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.ok(res instanceof Array);
                if (res instanceof Array && res.length > 0) {
                  releaseId = res[0].id;
                }
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:owner/:repo/releases/:id (getRelease)",  function(next) {
        if (!releaseId) {
            next();
            return;
        }
        client.releases.getRelease(
            {
                owner: owner,
                id: releaseId,
                repo: repo
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.equal(res.id, releaseId);
                next();
            }
        );
    });

    it("should successfully execute POST /repos/:owner/:repo/releases (createRelease)",  function(next) {
        if (!haveWriteAccess) {
          next();
          return;
        }
        client.releases.createRelease(
            {
                owner: owner,
                repo: repo,
                tag_name: "node-github-tag",
                target_commitish: "master",
                name: "node-github-name",
                body: "node-github-body",
                draft: false,
                prerelease: true,
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.equal(res.tag_name, "node-github-tag");
                Assert.equal(res.target_commitish, "master");
                Assert.equal(res.name, "node-github-name");
                Assert.equal(res.body, "node-github-body");
                Assert.equal(res.assets.length, 0);
                Assert.ok(res.prerelease);
                Assert.ok(!res.draft);
                newReleaseId = res.id;
                next();
            }
        );
    });

    it("should successfully execute PATCH /repos/:owner/:repo/releases/:id (editRelease)",  function(next) {
        if (!haveWriteAccess) {
          next();
          return;
        }
        client.releases.editRelease(
            {
                owner: owner,
                id: newReleaseId,
                repo: repo,
                tag_name: "node-github-new-tag",
                target_commitish: "master",
                name: "node-github-new-name",
                body: "node-github-new-body",
                draft: true,
                prerelease: true,
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.equal(res.id, newReleaseId);
                Assert.equal(res.tag_name, "node-github-new-tag");
                Assert.equal(res.target_commitish, "master");
                Assert.equal(res.name, "node-github-new-name");
                Assert.equal(res.body, "node-github-new-body");
                Assert.equal(res.assets.length, 0);
                Assert.ok(res.prerelease);
                Assert.ok(res.draft);
                next();
            }
        );
    });

    it("should successfully execute DELETE /repos/:owner/:repo/releases/:id (deleteRelease)",  function(next) {
        if (!haveWriteAccess) {
          next();
          return;
        }
        client.releases.deleteRelease(
            {
                owner: owner,
                repo: repo,
                id: newReleaseId,
            },
            function(err, res) {
                Assert.equal(err, null);
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:owner/:repo/releases/:id/assets (listAssets)",  function(next) {
        client.releases.listAssets(
            {
                owner: owner,
                id: releaseIdWithAsset,
                repo: repo
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.ok(res instanceof Array);
                if (res instanceof Array && res.length > 0) {
                    assetId = res[0].id;
                }
                next();
            }
        );
    });

    it("should successfully execute GET /repos/:owner/:repo/releases/assets/:id (getAsset)",  function(next) {
        if (!assetId) {
            next();
            return;
        }
        client.releases.getAsset(
            {
                owner: owner,
                id: assetId,
                repo: repo
            },
            function(err, res) {
                Assert.equal(err, null);
                Assert.equal(res.id, assetId);
                next();
            }
        );
    });

    it("should successfully execute PATCH /repos/:owner/:repo/releases/assets/:id (editAsset)",  function(next) {
        if (!newAssetId) {
            next();
            return;
        }
        client.releases.editAsset(
            {
                owner: owner,
                id: "Number",
                repo: repo,
                name: "String",
                label: "String"
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });

    it("should successfully execute DELETE /repos/:owner/:repo/releases/assets/:id (deleteAsset)",  function(next) {
        if (!newAssetId) {
            next();
            return;
        }
        client.releases.deleteAsset(
            {
                owner: owner,
                id: "Number",
                repo: repo
            },
            function(err, res) {
                Assert.equal(err, null);
                // other assertions go here
                next();
            }
        );
    });
});
