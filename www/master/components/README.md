Components
==========

A tab in the Kubernetes UI with its set of visualizations is referred to as a *component*. Components are separated from the chrome and services to simplify the development of new visualizations. This document describes how to create and modify components.

Each component has its own directory in `www/master/components`. The component directory contains a manifest file and the files comprising the component, such as HTML  pages and views, Angular providers, CSS and Less files, and other assets. Here is the recommended structure for a component directory:

```
foo_component
├── config
├── css
├── img
├── js
│   └── modules
│       ├── controllers
│       ├── directives
│       └── services
├── less
├── pages
├── protractor
├── test
├── views
│   └── partials
└── manifest.json
```

###Manifest file

The JSON-formatted manifest file, named `manifest.json`, resides at the root of the directory. Using the component directory name and the contents of the manifest, the Kubernetes UI automatically adds a tab to the chrome, a dependency on the component's AngularJS module in the main AngularJS app, and Angular routes for the component.

For example, consider the following manifest file at `master/components/foo_component/manifest.json`:

```
{
  "routes": [
    {
      "url": "/",
      "templateUrl": "/components/foo_component/pages/home.html"
    },
    {
      "url": "/kittens",
      "templateUrl": "/components/foo_component/pages/kittens.html",
      "css": "/components/foo_component/css/kittens.css"
    }
  ]
}
```

From the name of the component directory, the Kubernetes UI:

* creates a tab called "Foo Component",
* adds Angular module `kubernetesApp.components.fooComponent` to the dependencies of `kubernetesApp`, and
* defines Angular routes `/foo_component/` and `/foo_component/kittens`.

Every tab links to `/` relative to its component, so it is important to always define a `/` route.

###Source files
Many of the files located in `master/components/<component>` are copied to `app/components/<component>/` on each gulp build. This includes (but is not limited to) HTML pages and views, CSS files and images.

Exceptions include the `config` and `less` directories, and all of the `.js` files. The following sections explain how the exceptions are built into the UI.

####JavaScript
All JavaScript files located in the `master/components/<component>/js` are concatenated together with the rest of the UI's JavaScript and written to `app/assets/js/app.js`. In the production build, they are also uglified.

####Configuration

Similar to the [application wide configuration](../../README.md#configuration), components can define environment specific configuration. The gulp task creates the constant `ENV` under the `kubernetesApp.config` module, which is an object with a property for the root UI and each component.

For example, a configuration for the `development` environment specific to `foo_component` would be located at `master/components/foo_component/config/development.json`. Such a component would access its `development.json` configuration verbatim at `ENV.foo_component`:

```
angular.module('kubernetesApp.components.fooComponent', ['kubernetesApp.config'])
    .provider('foo', ...)
    .config(['fooProvider', 'ENV', function(fooProvider, ENV) {
      // Configure fooProvider using ENV['foo_component'].
    });
```

####Less

Like JavaScript, the component's Less files are built into one monolithic CSS file. All top-level Less files located at `master/components/<component>/less/*.less` are imported into the main UI's Less file. The result is then copied to `app/assets/css/app.css`. In the production build, it is also minified.

Sub-directories of this path are watched for changes, but not directly imported. This is useful for defining common colors, mixins and other functions or variables used by the top-level Less files.

###Appendix

####Manifest schema

```
{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "type": "object",
  "properties": {
    "description": {
      "type": "string",
      "description": "Very brief summary of the component. Use a README.md file for detailed descriptions."
    },
    "routes": {
      "type": "array",
      "description": "Angular routes for the component.",
      "items": {
        "type": "object",
        "properties": {
          "description": {
            "type": "string",
            "description": "Short description of the route."
          },
          "url": {
            "type": "string",
            "description": "Route location relative to '/<component>'."
          },
          "templateUrl": {
            "type": "string",
            "description": "Absolute location of the HTML template."
          },
          "css": {
            "type": "string",
            "description": "Absolute location of CSS to use with this route."
          }
        },
        "required": ["url", "templateUrl"]
      },
      "minItems": 1
    }
  },
  "required": ["routes"]
}
```

[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/www/master/components/README.md?pixel)]()
