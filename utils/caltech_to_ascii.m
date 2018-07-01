arg_list = argv ();
if(nargin > 0)
    dir = arg_list{1};
    annotation_dir = [dir "/Annotations/**/*.mat"]
    file_list = glob(annotation_dir);
    for(i=1:length(file_list))
           load(file_list{i})
           new_path = strrep(strrep(file_list{i},"Annotations/","Annotations_txt/"),".mat",".txt")
           new_dir = fileparts(new_path)
           system(["mkdir -p " new_dir])
           save("-ascii", [new_path],"box_coord")
    end 
endif

