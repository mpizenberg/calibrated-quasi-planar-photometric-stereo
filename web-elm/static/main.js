// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

// Function returning the size of the container element for the app.
// In our case, the full layout viewport.
const layoutViewportSize = () => ({
  width: window.innerWidth,
  height: window.innerHeight,
});

// // Initialize the database with a "dependencies" store.
// const db = await idb.openDB('PubGrub', 1, {
// 	upgrade(db) { db.createObjectStore('dependencies'); }
// });
//
// // Retrieve all keys and values of stored dependencies.
// const keys = await db.getAllKeys('dependencies');
// const values = await db.getAll('dependencies');

// // Using local storage to hold the session.
// var storageKey = "session";
// var flags = localStorage.getItem(storageKey);

// Start the Elm application.
var app = Elm.Main.init({
  node: document.getElementById("app"),
  // flags: { keys: keys, values: values },
  flags: layoutViewportSize(),
});

// Activate the app ports.
import { activatePorts } from "./ports.js";
activatePorts(app, layoutViewportSize);
