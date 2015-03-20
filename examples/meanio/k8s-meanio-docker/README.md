[![Build Status](https://travis-ci.org/linnovate/mean.svg?branch=master)](https://travis-ci.org/linnovate/mean)
[![Dependencies Status](https://david-dm.org/linnovate/mean.svg)](https://david-dm.org/linnovate/mean)
[![Gitter](https://badges.gitter.im/JoinChat.svg)](https://gitter.im/linnovate/mean?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# [![MEAN Logo](http://mean.io/system/assets/img/logos/meanlogo.png)](http://mean.io/) MEAN Stack

MEAN is a framework for an easy starting point with [MongoDB](http://www.mongodb.org/), [Node.js](http://www.nodejs.org/), [Express](http://expressjs.com/), and [AngularJS](http://angularjs.org/) based applications. It is designed to give you a quick and organized way to start developing MEAN based web apps with useful modules like Mongoose and Passport pre-bundled and configured. We mainly try to take care of the connection points between existing popular frameworks and solve common integration problems.

## Prerequisite Technologies
### Linux
* *Node.js* - <a href="http://nodejs.org/download/">Download</a> and Install Node.js, nodeschool has free <a href=" http://nodeschool.io/#workshoppers">node tutorials</a> to get you started.
* *MongoDB* - <a href="http://www.mongodb.org/downloads">Download</a> and Install mongodb - <a href="http://docs.mongodb.org/manual">Checkout their manual</a> if you're just starting.

If you're using ubuntu this is the preffered repository to use...

```bash
$ curl -sL https://deb.nodesource.com/setup | sudo bash -
$ sudo apt-get update
$ sudo apt-get install nodejs
```

* *Git* - Get git using a package manager or <a href="http://git-scm.com/downloads">download</a> it.

### Windows
* *Node.js* - <a href="http://nodejs.org/download/">Download</a> and Install Node.js, nodeschool has free <a href=" http://nodeschool.io/#workshoppers">node tutorials</a> to get you started.
* *MongoDB* - Follow the great tutorial from the mongodb site - <a href="http://docs.mongodb.org/manual/tutorial/install-mongodb-on-windows">"Install Mongodb On Windows"</a>
* *Git* - The easiest way to install git and then run the rest of the commands through the *git bash* application is by downloading and installing <a href="http://git-scm.com/download/win">Git for Windows</a>

### OSX
* *Node.js* -  <a href="http://nodejs.org/download/">Download</a> and Install Node.js or use the packages within brew or macports.
* *MongoDB* - Follow the tutorial here - <a href="http://docs.mongodb.org/manual/tutorial/install-mongodb-on-os-x/">Install mongodb on OSX</a>
* *git* - Get git <a href="http://git-scm.com/download/mac">from here</a>.

## Prerequisite packages

* Mean currently works with either grunt or gulp..
```
$ npm install -g gulp
// and bower
$ npm install -g bower 
```

## Installation
To start with MEAN install the `mean-cli` package from NPM.
This will add the *mean* command which lets you interact (install, manage, update ...) your Mean based application.

### Install the MEAN CLI

```bash
$ npm install -g mean-cli
$ mean init <myApp>
$ cd <myApp> && npm install
```

### Invoke node with a task manager
Mean supports the gulp task runner for various services which are applied on the code.
To start you application run - 
```bash
$ gulp
```

Alternatively, when not using `gulp` (and for production environments) you can run:
```bash
$ node server
```
Then, open a browser and go to:
```bash
http://localhost:3000
```

### Troubleshooting
During installation depending on your os and prerequiste versions you may encounter some issues.

Most issues can be solved by one of the following tips, but if you are unable to find a solution feel free to contact us via the repository issue tracker or the links provided below.

#### Update NPM, Bower or Grunt
Sometimes you may find there is a weird error during install like npm's *Error: ENOENT*. Usually updating those tools to the latest version solves the issue.

* Updating NPM:
```bash
$ npm update -g npm
```

* Updating Grunt:
```bash
$ npm update -g grunt-cli
```

* Updating Bower:
```bash
$ npm update -g bower
```

#### Cleaning NPM and Bower cache
NPM and Bower has a caching system for holding packages that you already installed.
We found that often cleaning the cache solves some troubles this system creates.

* NPM Clean Cache:
```bash
$ npm cache clean
```

* Bower Clean Cache:
```bash
$ bower cache clean
```

#### Installation problems on Windows 8 / 8.1
Some of Mean.io dependencies uses [node-gyp](https://github.com/TooTallNate/node-gyp) with supported Python version 2.7.x. So if you see an error related to node-gyp rebuild follow next steps:

1. install [Python 2.7.x](https://www.python.org/downloads/)
2. install [Microsoft Visual Studio C++ 2012 Express](http://www.microsoft.com/ru-ru/download/details.aspx?id=34673)
3. Run NPM update

```bash
$ npm update -g
```

## Technologies

### The MEAN stack

MEAN is an acronym for *M*ongo, *E*xpress.js , *A*ngular.js and *N*ode.js

<dl class="dl-horizontal">
<dt>MongoDB</dt>
<dd>Go through MongoDB Official Website and proceed to its Great Manual, which should help you understand NoSQL and MongoDB better.</dd>
<dt>Express</dt>
<dd>The best way to understand express is through its Official Website, particularly The Express Guide; you can also go through this StackOverflow thread for more resources.</dd>
<dt>AngularJS</dt>
<dd>Angular's Official Website is a great starting point. CodeSchool and google created a <a href="https://www.codeschool.com/courses/shaping-up-with-angular-js">great tutorial</a> for beginners, and the angular videos by <a href="https://egghead.io/">Egghead</a>.</dd>
<dt>Node.js</dt>
<dd>Start by going through Node.js Official Website and this StackOverflow thread, which should get you going with the Node.js platform in no time.</dd>
</dl>

### Additional Tools
* <a href="http://mongoosejs.com/">Mongoose</a> - The mongodb node.js driver in charge of providing elegant mongodb object modeling for node.js
* <a href="http://passportjs.org/">Passport</a> - An authentication middleware for Node.js which supports authentication using a username and password, Facebook, Twitter, and more.
* <a href="http://getbootstrap.com/">Twitter Bootstrap</a> - The most popular HTML, CSS, and JS framework for developing responsive, mobile first projects.
* <a href="http://angular-ui.github.io/bootstrap/">UI Bootstrap</a> - Bootstrap components written in pure AngularJS


## CLI
### Overview

The MEAN CLI is a simple Command Line Interface for installing and managing MEAN applications. As a core module of the Mean.io project, it provides a number of useful tools to make interaction with your MEAN application easier, with features such as: scaffolding, module creation and admin, status checks, and user management.
```bash
$ mean
$ mean --help
$ mean help
```
  <code>mean help</code> can also be used in conjunction with any command to get more information about that particular functionality. For example, try <code>mean help init</code> to see the options for init
```bash
$ mean help [command]
```
### Users

 <p>Information can be display for a specific customer via <code>mean user email</code>. Email is required. User roles can be assigned or removed with the <code>--addRole (or -a)</code> and <code>--removeRole (or -r)</code> options, respectively.</p>
  <p>For example, the <i>admin</i> role is required to edit tokens.</p>

```bash
$ mean user <email>
$ mean user <email> --addRole <role>;
$ mean user <email> --removeRole <role>;
```

### Packages
#### Management
 <p class="alert alert-warning">All of the remaining of the commands must be run from the root folder of your MEAN application.</p>
  <p>Contributed MEAN packages can be installed or uninstalled via the CLI. Also, currently installed modules can be viewed with the <code>list</code> command.</p>

```bash
$ mean list
$ mean install <module>
$ mean uninstall <module>
```

  <p class="alert alert-info">Mean packages installed via the installer are found in <i>/node_modules</i></p>
#### Search
To find new packages run the *mean search* command
```bash
$ mean search [packagename]
```
`mean search` will return all of the available packages, `mean search [packagename]` will filter the search results.

#### Scaffolding
To create a new MEAN app, run <code>mean init</code>. Name for the application is optional. If no name is provided, "mean" is used. The MEAN project will be cloned from GitHub into a directory of the application name.
```bash
$ mean init [name]
$ cd [name] && npm install
```
  <p class="alert alert-info">Note: <a href="http://git-scm.com/downloads">git</a> must be installed for this command to work properly.</p>

### Misc
<h4>Status</h4>
<p>Check the database connection for a particular environment (e.g. development (default), test, production) and make sure that the meanio command line version is up to date.</p>
```bash
$ mean status
```
<h4>Docs</h4>
<p>A simple shortcut to open the mean documentation in your default browser.</p>
```bash
$ mean docs
```

## Packages

Everything in mean.io is a package and when extending mean with custom functionality make sure you create your own package and do not alter the core packages.

The mean.io package system allows developers to create modular code that provides useful tools that other mean developers can use. The packages, when published, are plug-and-play and are used in a way very similar to traditional npm packages.

The mean.io package system integrates all the packages into the mean project as if the code was part of mean itself and provides the developers with all the necceesary tools required to integrate their package into the host project.

There are two types of packages:

**Custom Packages** are generated by the mean scaffolder and contain most of your application logic. Custom packages are found in */packages/custom* and can be published as a contrib package for use by other developers.

**Contrib Packages** are installed by the mean installer and are found at */packages/contrib*. Contrib packages are "plug and play".

### Core Packages

All `Core` packages can be overridden by other packages allowing you to extend and adapt it to fit your specific needs. See `Overriding views` for detailed examples.


#### System
The "system" package creates the basic pages as well as defines the layout of the site and integrates the menu into the page. The system package also allows us to define things such as rendering engines, static files and routing on the client and server side.
#### Users
The "users" package creates the database model of the user, provides validation as well as various login and registration features.
#### Access
The "access" package managers permissions and middleware. It controls the various authentication methods and is dependent on the users package
#### Theme
The "theme" package adds some basic CSS and other assets such as images and backgrounds
#### Articles
The "articles" package is typically used as an example starting point for managing content that might be used in a blog or cms. The full CRUD is implemented on the server and client.
### Files structure
The file structure is similar to that of the mean project itself

`Fundamental` Files at the `root` of the package

**Server**

Packages are registered in the **app.js** 
Defines package name, version and `mean=true` in the **package.json**   

All of the Server side code resides in the `/server` directory.

    Server
    --- config        # Configuration files
    --- controllers   # Server side logic goes here
    --- models        # Database Schema Models
    --- routes        # Rest api endpoints for routing
    --- views         # Swig based html rendering

**Client**

All of the Client side code resides in the `/public` directory.

    public            
    --- assets        # Javascript/Css/Images (not aggregated)
    --- controllers   # Angular Controllers
    --- config        # Contains routing files
    --- services      # Angular Services (also directive and filter folders)
    --- views         # Angular views

All javascript within public is automatically aggregated with the exception of files in assets which can be manually added using the `aggregateAsset()` function

Files within public of the package can be accessed externally `/[package-name]/path-to-file-relative-to-public` for example to access tokens angular controller tokens/controllers/tokens.js

###Registering a Package

In order for a Package to work it needs to be registered. By doing this you make package system aware that you are ready and that other packages are able to depend on you. The packages are registered from within `app.js` 

When registering you are required to declare all your dependencies in order for the package system to make them available to your package.

```javascript
// Example of registering the MyPackage
MyPackage.register(function(app, auth, database) {
  // ...

});
```

MEAN has 3 pre registered dependencies:
  - `app` Makes the express app available .
  - `auth` Includes some basic authentication functions
  - `database` Contains the Mongoose database connection

> All dependencies specified must be registered in order to use them

###Dependency Injection

> An injection is the passing of a dependency (a service) to a dependent
> object (a client). The service is made part of the client's state.
> Passing the service to the client, rather than allowing a client to
> build or find the service, is the fundamental requirement of the
> pattern. [Wikipedia](http://en.wikipedia.org/wiki/Dependency_injection)


Dependency injection allows you to declare what dependencies you require and rely on the package system to resolve all dependencies for you. Any package registered is automatically made available to anyone who would like to depend on them.

Looking again at the registration example we can see that `MyPackage` depends on the `Tokens` and can make use of it full functionality including overriding it.
 
```javascript
// Example of registering the tokens package
MyPackage.register(function(app, auth, database, Tokens) {

  // I can make use of the tokens within my module
  MyPackage.someExampleFunction('some parameter');

  // I can override functions
  MyPackage.someExampleFunction = function(param) {
    //my custom logic goes here
  };
});
```

> Packages when in code are used in a capitalized form

###Angular Modules and Dependencies

Every package registration automatically creates a corresponding angular module of the form `mean.[package-name]`

The package system injects this information into the mean init functions and allows developers to base their controllers, services, filters, directives etc on either an existing module or on their own one.

In addition you are able to declare which angular dependencies you want your angular module to use.

Below is an example of adding an angular dependency to our angular module.

```javascript
// Example of adding an angular dependency of the ngDragDrop to the
MyPackage.angularDependencies(['ngDragDrop']);
```

> See the assets section for an example how to add external libraries to
> the client such as the `gDragDrop `javascript library

###Assets and Aggregation

All assets such as images, javascript libraries and CSS stylesheets should be within `public/assets` of the package file structure.

Javascript and CSS from `assets` can be aggregated to the global aggregation files. By default all javascript is automatically wrapped within an anonymous function unless given the option `{global:true}` to not enclose the javascript within a contained scope


```javascript

//Adding jquery to the mean project
MyPackage.aggregateAsset('js','jquery.min.js');

//Adding another library - global by default is false
MyPackage.aggregateAsset('js','jquery.min.js', {global:true});

//Adding some css to the mean project
MyPackage.aggregateAsset('css','default.css');
```

> Javascript files outside of assets are automatically aggregated and
> injected into the mean project. As a result libraries that you do not
> want aggregated should be placed within `public/assets/js`

The aggregation supports the ability to control the location of where to inject the aggregated code and if you add a weight and a group to your aggregateAsset method you can make sure it's included in the correct region.

```javascript
MyPackage.aggregateAsset('js','first.js',{global:true,  weight: -4, group: 'header'});
```

>The line that gets loaded in your head.html calls the header group and injects the js you want to include first-
> in packages/system/server/views/includes/head.html 
> <script type="text/javascript" src="/modules/aggregated.js?group=header"></script>

###Settings Object
The settings object is a persistance object that is stored in the packages collection and allows for saving persistant information per package such as configuration options or admin settings for the package.

  Receives two arguments the first being the settings object the second is a callback function
  
```javascript
MyPackage.settings({'someSetting':'some value'}, function (err, settings) {
    // You will receive the settings object on success
});

// Another save settings example this time with no callback
// This writes over the last settings.
MyPackage.settings({'anotherSettings':'some value'});

// Get settings. Retrieves latest saved settings
MyPackage.settings(function (err, settings) {
  // You now have the settings object
});
```

> Each time you save settings you overwrite your previous value.
> Settings are designed to be used to store basic configuration options
> and should not be used as a large data store


###Express Routes
All routing to server side controllers is handled by express routes. The package system uses the typical express approach. The package system has a route function that passes along the package object to the main routing file typically `server/routes/myPackage.js`

  By default the Package Object is passed to the routes along with the other arguments
  `MyPackage.routes(app, auth, database);`


Example from the `server/routes/myPackage.js`

```javascript
// The Package is past automatically as first parameter
module.exports = function(MyPackage, app, auth, database) {

  // example route
  app.get('/myPackage/example/anyone', function (req,res,next) {
    res.send('Anyone can access this');
  });
};
```

###Angular Routes
The angular routes are defined in `public/routes/myPackage.js`. Just like the latest version of mean, the packages  use the `$stateProvider`

```javascript
$stateProvider
  .state('myPackage example page', {
    url: '/myPackage/example',
    templateUrl: 'myPackage/views/index.html'
  });
```

> The angular views are publically accessible via templateUrl when
> prefixed with the package name

###Menu System

Packages are able to hook into an existing menu system and add links to various menus integrated within Mean.

Each link specifies its `title`, `template`, `menu` and `role` that is allowed to see the link. If the menu specified does not exist a new menu will be created. The menu object is made accessible within the client by means of a *menu angular service* that queries the menu controller for information about links.

Below is an example how to add a link to the main menu from `app.js`

```javascript
//We are adding a link to the main menu for all authenticated users
MyPackage.menus.add({
  title: "myPackage example page",
  link: "myPackage example page",
  roles: ["authenticated"],
  menu: "main"
});
```


> You can look at the angular header controller in the mean project for
> more info. You can find it `public/system/controllers/header.js` and
> see how the menu service is implemented

###Html View Rendering
The packages come built in with a rendering function allowing packages to render static html. The default templating engine is *swig*. The views are found in `server/views` of the package and should end with the `.html` suffix

Below is an example rendering some simple html

```javascript
app.get('/myPackage/example/render', function (req,res,next) {
  MyPackage.render('index', {packageName:'myPackage'}, function (err, html) {
    //Rendering a view from the Package server/views
    res.send(html);
  });
});
```

###Overriding the default layouts
One is able to override the default layout of the application through a custom package.

Below is an example overriding the default layout of system and instead using the layourts found locally within the package

```javascript
MyPackage.register(function(system, app) {
  app.set('views', __dirname + '/server/views');
  // ...
```

> Please note that the package must depend on `System` to ensure it is
> evaluated after `System` and can thus override the views folder

### Overriding views
You may override public views used by certain core packages.  To create a custom home page, you would create a custom package and modify the script in it's public folder like so:

```javascript
angular.module('mean.mycustompackage', ['mean.system'])
.config(['$viewPathProvider', function($viewPathProvider) {
  $viewPathProvider.override('system/views/index.html', 'mycustompackage/views/myhomepage.html');
}]);
```

This will render *mycustompackage/views/myhomepage.html* as the home page.

### Creating your own package
To create your own package and scaffold its initial code, run the following command:

```bash
$ mean package <packageName>
```

This will create a package under */packages/custom/pkgName*

### Deleting a package
To delete your package, and remove its files:

```bash
$ mean uninstall myPackage
```
Where "myPackage" is the name of your package.


### Contributing your package
Once your package is in good shape and you want to share it with the world you can start the process of contributing it and submiting it so it can be included in the package repository.
To contribute your package register to the network (see the section below) and run

```bash 
$ mean register # register to the mean network (see below)
$ cd <packages/custom/pkgName>
$ mean publish
```

## MEAN Network
Mean is a stand-alone instance that you can install locally or host on your server.
We want to provide value to developers and are assembling a set of services which will be called the mean network.
We're building all of this as we speak but we allready have some elements in place.

### Network User Management

#### Registration
```bash
$ mean register
```
#### Identity
```bash
$ mean whoami
```
### Deploy
Coming soon!

## Config
All the configuration is specified in the [config](/config/) folder,
through the [env](config/env/) files, and is orchestrated through the [meanio](https://github.com/linnovate/meanio) NPM module.
Here you will need to specify your application name, database name, and hook up any social app keys if you want integration with Twitter, Facebook, GitHub, or Google.

### Environmental Settings

There is a shared environment config: __all__

* __root__ - This the default root path for the application.
* __port__ - DEPRECATED to __http.port__ or __https.port__.
* __http.port__ - This sets the default application port.
* __https__ - These settings are for running HTTPS / SSL for a secure application.
* __port__ - This sets the default application port for HTTPS / SSL. If HTTPS is not used then is value is to be set to __false__ which is the default setting. If HTTPS is to be used the standard HTTPS port is __443__.
* __ssl.key__ - The path to public key.
* __ssl.cert__ - The path to certificate.

There are three environments provided by default: __development__, __test__, and __production__.
Each of these environments has the following configuration options:

* __db__ - This is where you specify the MongoDB / Mongoose settings
* __url__ - This is the url/name of the MongoDB database to use, and is set by default to __mean-dev__ for the development environment.
* __debug__ - Setting this option to __true__ will log the output all Mongoose executed collection methods to your console.  The default is set to __true__ for the development environment.
* __options__ - These are the database options that will be passed directly to mongoose.connect in the __production__ environment: [server, replset, user, pass, auth, mongos] (http://mongoosejs.com/docs/connections.html#options) or read [this] (http://mongodb.github.io/node-mongodb-native/driver-articles/mongoclient.html#mongoclient-connect-options) for more information.
* __app.name__ - This is the name of your app or website, and can be different for each environment. You can tell which environment you are running by looking at the TITLE attribute that your app generates.
* __Social OAuth Keys__ - Facebook, GitHub, Google, Twitter. You can specify your own social application keys here for each platform:
  * __clientID__
  * __clientSecret__
  * __callbackURL__
* __emailFrom__ - This is the from email address displayed when sending an email.
* __mailer__ - This is where you enter your email service provider, username and password.

To run with a different environment, just specify NODE_ENV as you call grunt:
```bash
$ NODE_ENV=test grunt
```
If you are using node instead of grunt, it is very similar:
```bash
$ NODE_ENV=test node server
```
To simply run tests
```bash
$ npm test
```
> NOTE: Running Node.js applications in the __production__ environment enables caching, which is disabled by default in all other environments.

## Staying up to date
After initializing a project, you'll see that the root directory of your project is already a git repository. MEAN uses git to download and update its own code. To handle its own operations, MEAN creates a remote called `upstream`. This way you can use git as you would in any other project. 

To update your MEAN app to the latest version of MEAN

```bash
$ git pull upstream master
$ npm install
```

To maintain your own public or private repository, add your repository as remote. See here for information on [adding an existing project to GitHub](https://help.github.com/articles/adding-an-existing-project-to-github-using-the-command-line).

```bash
$ git remote add origin <remote repository URL>
$ git push -u origin master
```


## Hosting MEAN
Since version 0.4.2 MEAN provides a command to easily upload your app to the *mean cloud*.
To do so all you need to do is the following steps.

1. make sure you have a unique name for your app (not the default mean) and that the name is in the package.json
1. Run `mean deploy`
1. It will create the meanio remote which can be used to update your remote app by `git push meanio master`
1. You can add remote command using the --remote flag for instance to add a role to a user on the remote cloud instance run `mean user -a RoleName emailAddress --remote`
1. To get an idea of whats happening on the mean log (node.js based logging) run `mean logs -n 100` to get the last 100 lines...

### Heroku
Before you start make sure you have the [Heroku toolbelt](https://toolbelt.heroku.com/)
installed and an accessible MongoDB instance - you can try [MongoHQ](http://www.mongohq.com/)
which has an easy setup).

Add the db string to the production env in *server/config/env/production.js*.

```bash
$ git init
$ git add .
$ git commit -m "initial version"
$ heroku apps:create
$ heroku config:add NODE_ENV=production
$ heroku config:add BUILDPACK_URL=https://github.com/mbuchetics/heroku-buildpack-nodejs-grunt.git
$ git push heroku master
$ heroku config:set NODE_ENV=production
```

### OpenShift

1. Register for an account on Openshift (http://www.openshift.com).
1. Create an app on Openshift by choosing a 'Node' type site to create. Create the site by making Openshift use Linnovate's Openshift git repo as its source code (https://github.com/linnovate/mean-on-openshift.git).
1. On the second screen after the new application has been created, add a Mongo database.
1. When the site has been built, you can visit it on your newly created domain, which will look like my-domain.openshift.com. You may need to restart the instance on Openshift before you can see it. It will look like Mean.io boilerplate.
1. On your new app's console page on Openshift, make a note of the git repo where the code lives. Clone that repo to your local computer where your mean.io app codebase is.
1. Merge your completed local app into this new repo. You will have some conflicts, so merge carefully, line by line.
1. Commit and push the repo with the Openshift code back up to Openshift. Restart your instance on Openshift, you should see your site!


## More Information
  * Visit us at [Linnovate.net](http://www.linnovate.net/).
  * Visit our [Ninja's Zone](http://www.meanleanstartupmachine.com/) for extended support.

## Credits
  * To our awesome <a href="https://github.com/orgs/linnovate/teams/mean">core team</a> with help of our <a href="https://github.com/linnovate/mean/graphs/contributors">contributors</a> which have made this project a success.
  * <a href="https://github.com/vkarpov15">Valeri Karpov</a> for coining the term *mean* and triggering the mean stack movement.
  * <a href="https://github.com/amoshaviv">Amos Haviv</a>  for the creation of the initial version of Mean.io while working for us @linnovate.
  * <a href="https://github.com/madhums/">Madhusudhan Srinivasa</a> which inspired us with his great work.

## License
We believe that mean should be free and easy to integrate within your existing projects so we chose [The MIT License](http://opensource.org/licenses/MIT)

