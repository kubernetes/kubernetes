// split high-order numeric attributes into surrogate pairs
require(__dirname).test
  ( { xml : '<a>&#x1f525;</a>'
    , expect :
      [ [ 'opentag', { name: 'A', attributes: {}, isSelfClosing: false } ]
      , [ 'text', '\ud83d\udd25' ]
      , [ 'closetag', 'A' ]
      ]
    , strict : false
    , opt : {}
    }
  )
