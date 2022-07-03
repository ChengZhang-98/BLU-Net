// Action to convert a contour tif file to a mask tif file.
// Note that only the files ending with ".tif" are regarded as contour files.
// PNG files are ignored as they are regarded as original files without contours.
function convertContours2Mask(input_dir, input_file, output_dir){
	open(input_dir + input_file);
	run("To ROI Manager");
	roiManager("Deselect");
	roiManager("Combine");
	// run("Make Inverse");
	run("Create Mask");
	input_file_split = split(input_file, ".tif");
	input_file_name = input_file_split[0];
	// print(input_file_name);
	// why does the first letter "i" disappear?
	saveAs("Tiff", output_dir + "i" + input_file_name + "_mask.tif");
	close();
	selectWindow(input_file);
	close();
	close("ROI Manager");
}

// Select the directory to input contour files and the directory to output mask files.
input_dir = getDirectory("Input Directory");
output_dir = getDirectory("Output Directory");
print("Input directory: "+ input_dir);
print("Output directory: "+ output_dir);

// Iterate the input directory to do conversion.
list = getFileList(input_dir);
file_cnt = 0;
for(i=0;i<list.length;i++){
	if (endsWith(list[i],".tif")){
	}
		convertContours2Mask(input_dir, list[i], output_dir);
		file_cnt++;
	}
}
print("Done! " + toString(file_cnt) + " contour file(s) was processed.");