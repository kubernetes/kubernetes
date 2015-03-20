'use strict';

/*
 * Defining the Package
 */
var mean = require('meanio'),
  Module = mean.Module;
function MeanUserKlass () {
  Module.call(this, 'users');
  this.auth = null;
}
MeanUserKlass.prototype = Object.create(Module.prototype,{constructor:{
  value:MeanUserKlass,
  configurable: false,
  enumerable: false,
  writable: false
}});

var MeanUser = new MeanUserKlass();

/*
 * All MEAN packages require registration
 * Dependency injection is used to define required modules
 */
MeanUser.register(function(app, database, passport) {
  // This is for backwards compatibility
  MeanUser.auth =require('./authorization');
  require('./passport')(passport);

  mean.register('auth', MeanUser.auth);

  //We enable routing. By default the Package Object is passed to the routes
  MeanUser.routes(app, MeanUser.auth, database, passport);

  //We are adding a link to the main menu for all authenticated users
  // MeanUser.menus.add({
  //     title: 'meanUser example page',
  //     link: 'meanUser example page',
  //     roles: ['authenticated'],
  //     menu: 'main'
  // });

  MeanUser.angularDependencies(['mean.system']);

  /**
    //Uncomment to use. Requires meanio@0.3.7 or above
    // Save settings with callback
    // Use this for saving data from administration pages
    MeanUser.settings({
        'someSetting': 'some value'
    }, function(err, settings) {
        //you now have the settings object
    });

    // Another save settings example this time with no callback
    // This writes over the last settings.
    MeanUser.settings({
        'anotherSettings': 'some value'
    });

    // Get settings. Retrieves latest saved settings
    MeanUser.settings(function(err, settings) {
        //you now have the settings object
    });
    */

  return MeanUser;
});
