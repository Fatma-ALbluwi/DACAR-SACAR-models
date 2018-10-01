
% for DACAR 3,4,5,6 layers: 

function im_output = DACAR(im_input, model)

weight = model.weight;
bias = model.bias;
im_y = single(im_input);

%% DA-CAR3 ( L3 (9, 7, 5)( 64, 32, 1) ):

convfea = vl_nnconv(im_y,weight{1},bias{1}, 'Pad',4);
convfea = vl_nnrelu(convfea);

convfea = vl_nnconv(convfea,weight{2},bias{2}, 'Pad',3);
convfea = vl_nnrelu(convfea);

convfea = vl_nnconv(convfea,weight{3},bias{3}, 'Pad',2);
im_output = convfea;

%% DA-CAR4 ( L4 (9, 3, 3, 5)( 64, 32, 32, 1) ): 

% convfea = vl_nnconv(im_y,weight{1},bias{1}, 'Pad',4);
% convfea = vl_nnrelu(convfea);
% 
% convfea = vl_nnconv(convfea,weight{2},bias{2}, 'Pad',1);
% convfea = vl_nnrelu(convfea);
% 
% convfea = vl_nnconv(convfea,weight{3},bias{3}, 'Pad',1);
% convfea = vl_nnrelu(convfea);
% 
% convfea = vl_nnconv(convfea,weight{4},bias{4}, 'Pad', 2);
% im_output = convfea;

%% DA-CAR5 ( L5 (9, 5, 5, 5, 5)(32, 32, 32, 32, 1) )

% convfea = vl_nnconv(im_y,weight{1},bias{1}, 'Pad',4);
% convfea = vl_nnrelu(convfea);
% 
% convfea = vl_nnconv(convfea,weight{2},bias{2}, 'Pad',2);
% convfea = vl_nnrelu(convfea);
% 
% convfea = vl_nnconv(convfea,weight{3},bias{3}, 'Pad',2);
% convfea = vl_nnrelu(convfea);
% 
% convfea = vl_nnconv(convfea,weight{4},bias{4}, 'Pad',2);
% convfea = vl_nnrelu(convfea);
% 
% convfea = vl_nnconv(convfea,weight{5},bias{5}, 'Pad', 2);
% im_output = convfea;

end
