<?xml version="1.0"?>

<xsl:stylesheet 
    xmlns:xsl="http://www.w3.org/1999/XSL/Transform" 
    xmlns:ui="http://dummy.com/mynamespace"
	  xmlns="http://www.w3.org/1999/xhtml" 
	  xmlns:html="http://www.w3.org/1999/xhtml"
    version="1.0">

  <xsl:output method="html" 
      doctype-system="http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd"
      doctype-public="-//W3C//DTD XHTML 1.0 Strict//EN" indent="yes"/>

  <xsl:template match="ui:app">
    <form>
      <xsl:attribute name="action">
        <xsl:value-of select="@url" />
      </xsl:attribute>
      <xsl:apply-templates select="./*" />
    </form>
  </xsl:template>

  <xsl:template match="ui:someButton">
    <input type="submit">
      <xsl:attribute name="id">
        <xsl:value-of select="@id" />
      </xsl:attribute>
      <xsl:attribute name="name">
        <xsl:value-of select="@name" />
      </xsl:attribute>
      <xsl:attribute name="value">
        <xsl:value-of select="normalize-space(.)"/>
      </xsl:attribute>
    </input>
  </xsl:template>

</xsl:stylesheet>