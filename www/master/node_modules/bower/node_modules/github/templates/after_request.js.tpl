
            if (!ret)
                ret = {};
            if (!ret.meta)
                ret.meta = {};
            [<%headers%>].forEach(function(header) {
                if (res.headers[header])
                    ret.meta[header] = res.headers[header];
            });
