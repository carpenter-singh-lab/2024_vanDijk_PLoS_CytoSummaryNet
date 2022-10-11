SELECT JUMP_MOA_compound_platemap_with_metadata_csv.well_position, 
JUMP_MOA_compound_platemap_with_metadata_csv.broad_sample, 
JUMP_MOA_compound_platemap_with_metadata_csv.pert_iname, 
JUMP_MOA_compound_platemap_with_metadata_csv.pert_type, 
JUMP_MOA_compound_platemap_with_metadata_csv.control_type, 
JUMP_MOA_compound_platemap_with_metadata_csv.moa,
Image.PathName_CellOutlines, 
Image.PathName_IllumAGP, 
Image.PathName_IllumBrightfield, 
Image.PathName_IllumDNA, 
Image.PathName_IllumER, 
Image.PathName_IllumMito, 
Image.PathName_IllumRNA, 
Image.PathName_NucleiOutlines, 
Image.PathName_OrigAGP, 
Image.PathName_OrigBrightfield, 
Image.PathName_OrigDNA, 
Image.PathName_OrigER, 
Image.PathName_OrigMito, 
Image.PathName_OrigRNA,
Image.FileName_OrigAGP, 
Image.FileName_OrigBrightfield, 
Image.FileName_OrigDNA, 
Image.FileName_OrigER, 
Image.FileName_OrigMito, 
Image.FileName_OrigRNA,
Image.Width_OrigDNA,
Image.TableNumber, 
Image.ImageNumber, 
Image.Metadata_Plate,
Nuclei.Nuclei_Location_Center_X, 
Nuclei.Nuclei_Location_Center_Y
FROM Cytoplasm
INNER JOIN Nuclei ON Cytoplasm.TableNumber=Nuclei.TableNumber AND Cytoplasm.ObjectNumber=Nuclei.ObjectNumber
INNER JOIN Cells ON Cytoplasm.TableNumber=Cells.TableNumber AND Cytoplasm.ObjectNumber=Cells.ObjectNumber
INNER JOIN Image on Cytoplasm.TableNumber=Image.TableNumber 
INNER JOIN JUMP_MOA_compound_platemap_with_metadata_csv on Image.well=JUMP_MOA_compound_platemap_with_metadata_csv.well_position 
WHERE Image.well IN ()