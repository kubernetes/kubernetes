# Readme

URL: https://github.com/swagger-api/swagger-ui/tree/master/dist
License: Apache License, Version 2.0
License File: LICENSE

## Description
Files from dist folder of https://github.com/swagger-api/swagger-ui.
These are dependency-free collection of HTML, Javascript, and CSS assets that
dynamically generate beautiful documentation and sandbox from a
Swagger-compliant API.
Instructions on how to use these:
https://github.com/swagger-api/swagger-ui#how-to-use-it

## Local Modifications
- Updated the url in index.html to "../../swaggerapi" as per instructions at:
https://github.com/swagger-api/swagger-ui#how-to-use-it
- Modified swagger-ui.js to list resources and operations in sorted order: https://github.com/kubernetes/kubernetes/pull/3421
- Set supportedSubmitMethods: [] in index.html to remove "Try it out" buttons.
- Remove the url query param to fix XSS issue:
  https://github.com/kubernetes/kubernetes/pull/23234

LICENSE file has been created for compliance purposes.
Not included in original distribution.
