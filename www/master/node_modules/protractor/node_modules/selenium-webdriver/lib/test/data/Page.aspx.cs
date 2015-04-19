using System;
using System.Threading;

public partial class Page : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {
        Response.ContentType = "text/html";

        int lastIndex = Request.PathInfo.LastIndexOf("/");
        string pageNumber = (lastIndex == -1 ? "Unknown" : Request.PathInfo.Substring(lastIndex + 1));
        if (!string.IsNullOrEmpty(Request.QueryString["pageNumber"]))
        {
            pageNumber = Request.QueryString["pageNumber"];
        }
        Response.Output.Write("<html><head><title>Page" + pageNumber + "</title></head>");
        Response.Output.Write("<body>Page number <span id=\"pageNumber\">");
        Response.Output.Write(pageNumber);
        //Response.Output.Write("<script>var s=''; for (var i in window) {s += i + ' -> ' + window[i] + '<p>';} document.write(s);</script>")'
        Response.Output.Write("</span></body></html>");
    }
}
