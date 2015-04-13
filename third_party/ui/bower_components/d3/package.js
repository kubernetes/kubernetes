// Package metadata for Meteor.js.

Package.describe({
  name: "d3js:d3", // http://atmospherejs.com/d3js/d3
  summary: "D3 (official): A JavaScript visualization library for HTML and SVG.",
  version: "3.5.5",
  git: "https://github.com/mbostock/d3.git"
});

Package.onUse(function(api) {
  api.versionsFrom(["METEOR@1.0"]);
  api.addFiles("d3.js", "client");
});
