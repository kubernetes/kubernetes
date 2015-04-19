
/**
 * Export transports.
 */

module.exports = {
    websocket: require('./websocket')
  , flashsocket: require('./flashsocket')
  , htmlfile: require('./htmlfile')
  , 'xhr-polling': require('./xhr-polling')
  , 'jsonp-polling': require('./jsonp-polling')
};
