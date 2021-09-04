# LordOfRings
A method to efficiently identify rings in a sparse matrix.

## How it works
Given a sparse matrix containing rings as 1 values like:<br/>
o &nbsp; o &nbsp;    o    &nbsp;    o    &nbsp;    o    &nbsp;    o    &nbsp; o &nbsp; o <br />
o &nbsp; o &nbsp;    o    &nbsp;    o    &nbsp;    o    &nbsp;    o    &nbsp; o &nbsp; o <br />
o &nbsp; o &nbsp;    o    &nbsp;    o    &nbsp;    o    &nbsp;    o    &nbsp; o &nbsp; o <br />
o &nbsp; o &nbsp;    o    &nbsp;  **1**  &nbsp;  **1**  &nbsp;    o    &nbsp; o &nbsp; o <br />
o &nbsp; o &nbsp;  **1**  &nbsp;    o    &nbsp;    o    &nbsp;  **1**  &nbsp; o &nbsp; o <br />
o &nbsp; o &nbsp;  **1**  &nbsp;    o    &nbsp;    o    &nbsp;  **1**  &nbsp; o &nbsp; o <br />
o &nbsp; o &nbsp;    o    &nbsp;  **1**  &nbsp;  **1**  &nbsp;    o    &nbsp; o &nbsp; o <br />
o &nbsp; o &nbsp;    o    &nbsp;    o    &nbsp;    o    &nbsp;    o    &nbsp; o &nbsp; o <br />

The module LordOfRings evaluates radius and center of each ring using GPU <br/>
acceleration.

A semi-parallel Taubin algorithm has been developed to fit the single circle. <br/>
The Ptolemy theorem (about quadrilaterals inscribed in a circle) has been <br/>
applied to separate points that belong to different rings.

You can find more info and tutorials in the [documentation](https://lordofrings.readthedocs.io/en/latest/?badge=latest).

[![Build Status](https://travis-ci.com/Sara-a-r/LordOfRings.svg?branch=main)](https://travis-ci.com/Sara-a-r/LordOfRings)
[![Documentation Status](https://readthedocs.org/projects/lordofrings/badge/?version=latest)](https://lordofrings.readthedocs.io/en/latest/?badge=latest)
