
## Building docs site locally
Make sure mkdocs is installed

`pip install mkdocs`

Install the requirements of the docs:

`pip install -r docs/requirements.txt`

Run mkdoc:

`mkdocs serve`

## Setting EccoJS development environment

1. Clone the eccoJS repository 
2. Install eccoJS dependencies by running `npm install`
3. Run rollup so it bundles a new version of eccoJS whenevver we change the files
```
rollup --config .\rollup.config.js --watch
```
4. eccoJS is bundled into a file (dist/ecco-bundle.min.js). Due to CORS protections in browsers, we'll have to run a python web server to serve the file locally.
```
python3 -m http.server
```
5.In ecco/html/setup.html, point the ecco_url towards a local path:
```
    //var ecco_url = 'https://storage.googleapis.com/ml-intro/ecco/'
    var ecco_url = 'http://localhost:8000/'
```
6. Whenever we build, copy ecco-bundle.min.js to wherever the server is pointing and ecco_url describes. (for convience, consider changing the file: in eccojs/rollup.config.js so the file is exported directly to where the python web server is serving from)

## Developing for EccoJS
EccoJS is mostly used inside Jupyter notebooks. The development process, however, is better done in a javascript test file to begin with. Only when that functionality is ready should we proceed to integrate it with python and Jupyter. 


The Javascript test files are in `eccoJS/test`. Some make assertions on Javascript code functionality, and some tests actually build a visualization in an HTML page for visual inspection. The automated tests run against the node.js bundle of Ecco.

Tests are run using the `tape` command. You can run a specific test file using the command `tape <file name>'.
