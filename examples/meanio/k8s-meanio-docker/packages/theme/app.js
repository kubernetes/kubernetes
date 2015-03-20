'use strict';

/*
 * Defining the Package
 */
var Module = require('meanio').Module;

var Theme = new Module('theme');

/*
 * All MEAN packages require registration
 * Dependency injection is used to define required modules
 */
Theme.register(function(app, auth, database) {

  //We enable routing. By default the Package Object is passed to the routes
  Theme.routes(app, auth, database);

  //We are adding a link to the main menu for all authenticated users
//  Theme.menus.add({
//    title: 'theme example page',
//    link: 'theme example page',
//    roles: ['authenticated'],
//    menu: 'main'
//  });

  Theme.aggregateAsset('css', 'loginForms.css');
  Theme.aggregateAsset('css', 'theme.css');
  Theme.angularDependencies(['mean.system']);

  /**
    //Uncomment to use. Requires meanio@0.3.7 or above
    // Save settings with callback
    // Use this for saving data from administration pages
    Theme.settings({
        'someSetting': 'some value'
    }, function(err, settings) {
        //you now have the settings object
    });

    // Another save settings example this time with no callback
    // This writes over the last settings.
    Theme.settings({
        'anotherSettings': 'some value'
    });

    // Get settings. Retrieves latest saved settings
    Theme.settings(function(err, settings) {
        //you now have the settings object
    });
    */

  return Theme;
});
