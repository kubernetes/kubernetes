'use strict';

var mean = require('meanio');

module.exports = function(System, app, auth, database) {

  app.route('/admin/menu/:name')
    .get(function(req, res) {
      var roles = req.user ? req.user.roles : ['anonymous'];
      var menu = req.params.name || 'main';
      var defaultMenu = req.query.defaultMenu || [];

      if (!Array.isArray(defaultMenu)) defaultMenu = [defaultMenu];

      var items = mean.menus.get({
        roles: roles,
        menu: menu,
        defaultMenu: defaultMenu.map(function(item) {
          return JSON.parse(item);
        })
      });

      res.json(items);
    });
};
