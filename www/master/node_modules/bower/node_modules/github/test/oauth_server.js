"use strict";

var GitHubApi = require("./../index");

var github = new GitHubApi({
    version: "3.0.0"
});

/*
!!!UNCOMMENT THE FOLLOWING TO MAKE THIS SCRIPT WORK, BUT CHANGE THE CREDENTIALS TO YOUR OWN!!!
github.authenticate({
    type: "basic",
    username: "mikedeboertest",
    password: "test1324"
});
*/

github.authorization.getAll({}, function(err, res) {
    if (err)
        throw err;

    var ids = res.map(function(app) { return app.id; });

    function next(id) {
        github.authorization["delete"]({
            id: id
        }, function(err, res) {
            if (err)
                throw err;
            if (ids.length)
                next(ids.shift());
            else
                allRemoved();
        });
    }

    next(ids.shift());

    function allRemoved() {
        github.authorization.create({
            scopes: ["user", "public_repo", "repo", "repo:status", "delete_repo", "gist"],
            note: "Authorization created to create unit tests",
            note_url: "https://github.com/ajaxorg/node-github"
        }, function(err, res) {
            if (err)
                throw err;

            // you can use the token within server-side apps.
            // use it by doing:
            // github.authenticate({
            //     type: "oauth",
            //     token: e5a4a27487c26e571892846366de023349321a73
            // });
            console.log("TOKEN:", res.token);
            console.log(res);
        });
    }

});