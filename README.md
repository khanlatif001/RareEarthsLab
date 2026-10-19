# RareEarthsLab
A web-based platform for rare-earth spectroscopy, automated emission peak detection, spectral labeling, and electronic transition identification using reference energy-level databases.

The Streamlit sidebar includes plot appearance controls for peak-label and
x/y-axis-label and tick-label font sizes, spectrum line width, smoothed-signal
color, peak transition/position label visibility, plus several
intensity-background colormaps. The annotated plot can be
downloaded as a 300 dpi PNG sized 7 x 4.5 inches for manuscript use.
When the displayed x-axis is wavelength, editable Emin (nm) and Emax (nm)
fields show the uploaded spectrum's wavelength bounds by default. Changing
either field immediately updates the displayed and downloadable plots.

The intensity-background colormap menu includes emission-spectrum palettes
(Inferno, Magma, Plasma, Viridis, Turbo, Hot and Cividis), diverging comparison
palettes (Coolwarm, Seismic, RdYlBu and Spectral), intensity-map palettes
(YlGnBu, YlOrRd, Purples, Ocean, Terrain and Gist Earth), and specialized
palettes (Rainbow, HSV, Cool, Twilight and Twilight Shifted). Inferno remains
the default; diverging maps are intended for positive/negative comparisons,
while Twilight maps are intended for cyclic data.
