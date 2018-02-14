Cryptocurrency Analysis
===============================================================================

## Description
Machine Learning application for Cryptocurrency analysis running as a web application under the Flask framework.

## Installation

Requires Python 2.7 and Flask 0.12.


## Directory structure

`setup.py` - This is the file that is invoked to start up a development server.
It gets a copy of the app from the package and runs it.
This won’t be used in production, but it will see a lot of mileage in development.

`config.py` - This file contains most of the configuration variables that the app needs.

`/instance/config.py` - This file contains configuration variables that shouldn't be in version control.
This includes things like API keys and database URIs containing passwords.
This also contains variables that are specific to this particular instance of your application.
For example, you might have DEBUG = False in config.py, but set DEBUG = True in instance/config.py on your local machine
for development. Since this file will be read in after config.py, it will override it and set DEBUG = True.

`/cryptonalysis/` - This is the package that contains the web application.
`/cryptonalysis/__init__.py`- This file initializes the application and brings together all of the various components.
`/cryptonalysis/views.py` - This is where the routes are defined.
It may be split into a package of its own (app/views/) with related views grouped together into modules.
`/cryptonalysis/models.py` - This is where you define the models of the application.
This may be split into several modules in the same way as views.py.
`/cryptonalysis/static/` - This directory contains the public CSS, JavaScript, images and other files that you want to make public
via the app. It is accessible from app.com/static/ by default.
`cryptonalysis/templates/` - This is where you’ll put the Jinja2 templates for the app.

## Authors

* Gerardo Figueroa [gerardo_ofc@yahoo.com](gerardo_ofc@yahoo.com)
