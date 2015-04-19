var client = require('../index').createClient()
  , client2 = require('../index').createClient()
  , assert = require('assert');

client.once('subscribe', function (channel, count) {
  client.unsubscribe('x');
  client.subscribe('x', function () {
    client.quit();
    client2.quit();
  });
  client2.publish('x', 'hi');
});

client.subscribe('x');
