% the SACAR CNN for 5 or 6 layers + one concatente layer:

function im_output = SACAR(im_input, model)

weight = model.weight;
bias = model.bias;
im_y = single(im_input);

%% the first layer
convfea1 = vl_nnconv(im_y, weight{1}, bias{1}, 'Pad', 4);
convfea1 = vl_nnrelu(convfea1);

%% the second layer
convfea2 = vl_nnconv(convfea1, weight{2}, bias{2}, 'Pad', 2);
convfea2 = vl_nnrelu(convfea2);

%% concatenated layer
convfea3 = vl_nnconcat({convfea1, convfea2});

%% mapping layer
convfea4 = vl_nnconv(convfea3, weight{3}, bias{3}, 'Pad', 2);
convfea4 = vl_nnrelu(convfea4);

%% mapping layer
convfea5 = vl_nnconv(convfea4, weight{4}, bias{4}, 'Pad', 2);
convfea5 = vl_nnrelu(convfea5);

%% for the last layer (in 5 layrs)
% convfea6 = vl_nnconv(convfea5, weight{5}, bias{5}, 'Pad', 2); 
% im_output = convfea6;


%% for the last layer (in 6 layrs)

% convfea6 = vl_nnconv(convfea5, weight{5}, bias{5}, 'Pad', 2);
% convfea6 = vl_nnrelu(convfea6);
% 
% convfea7 = vl_nnconv(convfea6, weight{6}, bias{6}, 'Pad', 2);  
% im_output = convfea7;

%% for the last layer (in 7 layrs)

convfea6 = vl_nnconv(convfea5, weight{5}, bias{5}, 'Pad', 2);
convfea6 = vl_nnrelu(convfea6);

convfea7 = vl_nnconv(convfea6, weight{6}, bias{6}, 'Pad', 2);
convfea7 = vl_nnrelu(convfea7);

convfea8 = vl_nnconv(convfea7, weight{7}, bias{7}, 'Pad', 2);  
im_output = convfea8;

%%