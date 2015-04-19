var helpers = require('./helpers');

if (!helpers.hasSvn()) {
    console.warn('#######################################################');
    console.warn('It is recommended you install svn for complete testing!');
    console.warn('#######################################################');
}

// Cleanup the uncaughtException added by the tmp module
// It messes with the mocha uncaughtException event to caught errors
// Please note that is the Resolver that calls tmp.setGracefulCleanup()
// so we need to require that before
require('../lib/core/resolvers/Resolver');
process.removeAllListeners('uncaughtException');

require('./core/resolvers/resolver');
require('./core/resolvers/urlResolver');
require('./core/resolvers/fsResolver');
require('./core/resolvers/gitResolver');
require('./core/resolvers/gitFsResolver');
require('./core/resolvers/gitRemoteResolver');
require('./core/resolvers/gitHubResolver');
require('./core/resolvers/svnResolver');
require('./core/resolverFactory');
require('./core/resolveCache');
require('./core/packageRepository');
require('./core/scripts');
require('./core/Manager');
require('./renderers/StandardRenderer.js');
require('./renderers/JsonRenderer.js');
require('./commands/index.js');
require('./util/index.js');
