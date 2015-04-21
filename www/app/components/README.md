Components
==========

A tab in the Kubernetes UI with its set of visualizations is referred to as a *component*. Components are separated from the UI chrome and base data providers to simplify the development of new visualizations. This document provides reference for creation and modification of components.

Each component has its own directory, which contains a manifest file, HTML views, Angular providers, CSS, Less and other assets. Below is the recommended directory structure for a component.
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
├── views
│   └── partials
└── manifest.json
```

###Manifest file

The JSON-formatted manifest file, named ```manifest.json```, is located at the root of a component. Based on the component directory name and the contents of the manifest, the Kubernetes UI automatically adds a tab to the chrome, a dependency on the component's AngularJS module to main AngularJS app and Angular routes for the component.

For example, consider a manifest file at ```master/components/foo_component/manifest.json```:
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

From the name of the component directory, the Kubernetes UI
* creates a tab called "Foo Component",
* adds Angular module ```kubernetesApp.components.fooComponent``` to the dependencies of ```kubernetesApp```, and
* defines Angular routes ```/foo_component/``` and ```/foo_component/kittens```.

Every tab links to ```/``` relative to its component, so it is important to always define a ```/``` route.

###Source files
In general, all files located in ```master/components/<component>``` are copied to ```app/components/<component>/``` on each gulp build. This includes (but is not limited to) HTML views, CSS and images. Exceptions to this copy are the ```config``` and ```less``` directories as well as all ```.js``` files.

The sections below describe how the exceptions are built into the UI.

####JavaScript
All JavaScript files located in the ```master/components/<component>/js``` are uglified and concatenated together with the rest of the UI's JavaScript. Once aggregated, the JavaScript file is minified and written to ```app/assets/js/app.js```.

####Configuration

Similar to the [UI-wide configuration](../../README.md#configuration), components can define different configuration for each environment. The gulp task creates the constant ```ENV``` under the ```kubernetesApp.config``` module for configuration, which is an object with a property for the root UI and each component.

For example, a configuration for the ```development``` environment specific to ```foo_component``` would be located at ```master/components/foo_component/config/development.json```. Such a component would access its ```development.json``` configuration verbatim at ```ENV.foo_component```:
```
angular.module('kubernetesApp.components.fooComponent', ['kubernetesApp.config'])
    .provider('foo', ...)
    .config(['fooProvider', 'ENV', function(fooProvider, ENV) {
      // Configure fooProvider using ENV['foo_component'].
    });
```

####Less

Like JavaScript, the component's Less files are built into one monolithic CSS file. All top-level Less files located at ```master/components/<component>/less/*.less``` are imported into the main UI's Less file. The result is then minified and copied to ```app/assets/css/app.css```.

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

Content available under the [CC-By 3.0
license](http://creativecommons.org/licenses/by/3.0/)
