#include "main.hpp"
#include "Dataset.hpp"
/* TODO: Please design your data structure carefully so that you can work with the given dataset
 *       in this assignment. The below structures are just some suggestions.
 */
struct kDTreeNode{

    vector<int> data;
    kDTreeNode *left;
    kDTreeNode *right;
    int label;

    kDTreeNode(vector<int> data, int label = 0, kDTreeNode *left = NULL, kDTreeNode *right = NULL){
        this->data = data;
        this->left = left;
        this->right = right;
        this->label = label;
    }

    friend ostream &operator<<(ostream &os, const kDTreeNode &node){

        os << "(";
        for (int i = 0; i < node.data.size(); i++)
        {
            os << node.data[i];
            if (i != node.data.size() - 1)
            {
                os << ", ";
            }
        }
        os << ")";
        return os;
    }
};

class kDTree
{
private:
    int k;
    kDTreeNode *root;
    int count;

public:
    kDTree(int k = 2){
        this->root = NULL;
        this->count = 0;
        this-> k = k;
    }

    void deleteTree(kDTreeNode* tmp){
        if(tmp){
            deleteTree(tmp->left);
            deleteTree(tmp->right);
            delete tmp;
        }
        tmp = NULL;
    }
    ~kDTree(){
        deleteTree(this->root);

        this->root = NULL;
        
        this-> k = 2;
        this->count = 0;
    }

    kDTreeNode* Reccopy(const kDTreeNode* tmp){
        if(!tmp) return NULL;

        kDTreeNode *newNode = new kDTreeNode(tmp->data);
        newNode->left = Reccopy(tmp->left);
        newNode->right = Reccopy(tmp->right);
        return newNode;
    }


    const kDTree &operator=(const kDTree &other){
        this->root = this->Reccopy(other.root);
        this->count = other.count;
        this->k = other.k;

        return *this;
    }

    kDTree(const kDTree &other){
        this->root = this->Reccopy(other.root);
        this->count = other.count;
        this->k = other.k;

    }



    void RecinorderTraversal(kDTreeNode* tmp) const{
        if(!tmp) return;

        RecinorderTraversal(tmp->left);
        cout<< *tmp << " ";
        RecinorderTraversal(tmp->right);

    }
    void inorderTraversal() const{
        this->RecinorderTraversal(this->root);

    }


    void RecpreorderTraversal(kDTreeNode* tmp) const{
        if(!tmp) return;

        cout<< *tmp<< " ";
        RecpreorderTraversal(tmp->left);
        RecpreorderTraversal(tmp->right);

    }
    void preorderTraversal() const{
        this->RecpreorderTraversal(this->root);
    }


    void RecpostorderTraversal(kDTreeNode* tmp) const{
        if(!tmp) return;

        RecpostorderTraversal(tmp->left);
        RecpostorderTraversal(tmp->right);
        cout<< *tmp << " ";

    }
    void postorderTraversal() const{
        this->RecpostorderTraversal(root);
    }



    int Recheight(kDTreeNode* tmp) const{
        if(!tmp) return 0;
        int left = Recheight(tmp->left);
        int right = Recheight(tmp->right);
        return 1 + max(left, right);
    }
    int height() const{
        return this-> Recheight(root);

    }


    int RecnodeCount(kDTreeNode* tmp) const{
        if(!tmp) return 0;
        return 1 + RecnodeCount(tmp->left) + RecnodeCount(tmp->right);
             
    }
    int nodeCount() const{
        return this->count;
           
    }


    int RecleafCount(kDTreeNode* tmp) const{
        
        if( !tmp) return 0;
        if(!(tmp->left) && !(tmp->right)) return 1;

        return RecleafCount(tmp->left) + RecleafCount(tmp->right);
    
    }
    int leafCount() const{
        return RecleafCount(root);
    }

    kDTreeNode* Recinsert(const vector<int> &point, kDTreeNode* tmp, int level){
        if(!tmp) return new kDTreeNode(point);

        int a = level % k;
        if( point[a] < tmp->data[a] ) tmp->left = Recinsert(point,tmp->left, level + 1);
        else tmp->right = Recinsert( point, tmp->right, level + 1);

        return tmp;
    }
    void insert(const vector<int> &point){
        this->root = this->Recinsert( point, root, 0);
        count++;
                         
    }

    kDTreeNode* findMin(kDTreeNode* tmp, int level, int alpha){
        if(!tmp) return NULL;
        int a = level % k;
        if( a == alpha){
            if( !tmp->left ) return tmp;
            kDTreeNode* minNode = findMin(tmp->left, level + 1, alpha);
            return minNode;
        }else{
            kDTreeNode* minNode = tmp;
            kDTreeNode* minLeft = findMin(tmp->left, level + 1, alpha);
            kDTreeNode* minRight = findMin(tmp->right, level + 1, alpha);

            if(minLeft && minLeft->data[alpha] < minNode->data[alpha]) minNode = minLeft;

            if(minRight && minRight ->data[alpha] < minNode->data[alpha]) minNode = minRight;
            
            return minNode;
        }
    }
    kDTreeNode* Recremove(const vector<int> &point, kDTreeNode* tmp, int level){
        if(!tmp) return NULL;
        int alpha = level % k;

        if(tmp->data == point){
            if(!tmp->left && !tmp->right ){
                delete tmp;
                tmp = NULL;
                count--;
                return NULL;
            }
            if(tmp->right ){
                kDTreeNode* minNode = findMin(tmp->right, level + 1, alpha);
                if(minNode ){
                    tmp->data = minNode->data;
                    tmp->right = Recremove(minNode->data, tmp->right,  level + 1);
                }
            }
            else{
                kDTreeNode* minNode = findMin(tmp->left, level + 1, alpha);
                if(minNode){
                    tmp->data = minNode->data;
                    tmp->right = tmp->left;
                    tmp->left = NULL;
                    tmp->right = Recremove(minNode->data, tmp->right, level + 1);
                }
            }
        }
        else if( point[alpha] < tmp->data[alpha]) tmp->left = Recremove(point, tmp->left,  level + 1);
        else tmp->right = Recremove(point,tmp->right,  level + 1); 

        return tmp;
    }
    
    void remove(const vector<int> &point){
        this->root = Recremove(point, root,  0);
               
    }


    bool Recsearch(const vector<int> &point, kDTreeNode* tmp, int level){
        if(!tmp) return 0;
        int a = level % k;
        if(point == tmp->data) return 1;
        else if(point[a] < tmp->data[a] ) return Recsearch(point, tmp->left, level + 1);
        else return Recsearch(point, tmp->right,  level + 1);

    }
    bool search(const vector<int> &point){
        return Recsearch(point, root,  0);

    }



    vector<vector<int>> Merge(const vector<vector<int>>& left, const vector<vector<int>>& right, int alpha){
        vector<vector<int>> res;
        int l = 0, r = 0;
        int l_size = left.size() , r_size = right.size();

        while(l < l_size && r < r_size){
            if(left[l][alpha] <= right[r][alpha]){
                res.push_back(left[l]);
                l++;

            }else{
                res.push_back(right[r]);
                r++;

            }
        }

        while(l < l_size){
            res.push_back(left[l]);
            l++;

        }
        while(r < r_size){
            res.push_back(right[r]);
            r++;

        }

        return res;
    }

    void Merge(vector<vector<int>>& pointList, vector<int> &label, const vector<vector<int>>& left, const vector<vector<int>>& right, const vector<int>& leftLabel, const vector<int>& rightLabel,int alpha, int leftsize, int rightsize){
        
        vector<vector<int>> res;
        vector<int> resLabel;
        int l = 0, r = 0;

        while(l < leftsize && r < rightsize){
            if(left[l][alpha] <= right[r][alpha]){
                res.push_back(left[l]);
                resLabel.push_back(leftLabel[l]);
                l++;

            }else{
                res.push_back(right[r]);
                resLabel.push_back(rightLabel[r]);
                r++;

            }
        }

        while(l < leftsize){
            res.push_back(left[l]);
            resLabel.push_back(leftLabel[l]);
            l++;

        }
        while(r < rightsize){
            res.push_back(right[r]);
            resLabel.push_back(rightLabel[r]);
            r++;

        }

        pointList = res;
        label = resLabel;
    }



    vector<vector<int>> merge_Sort(const vector<vector<int>>& pointList, int alpha, int size){
        if(size <= 1) return pointList;

        int mid = (size - 1) / 2;
        vector<vector<int>> left(pointList.begin(), pointList.begin() + mid + 1);
        vector<vector<int>> right(pointList.begin() + mid + 1 , pointList.end());

        left = merge_Sort(left, alpha, left.size());
        right = merge_Sort(right, alpha, right.size());

        return Merge(left, right, alpha);
    }

    void merge_Sort( vector<vector<int>>& pointList, vector<int> &label, int alpha, int size){
        if(size <= 1) return;

        int mid = (size - 1) / 2;  

        vector<vector<int>> left(pointList.begin(), pointList.begin() + mid + 1 );
        vector<vector<int>> right(pointList.begin() + mid + 1 , pointList.end());

        vector<int> left_Label(label.begin(), label.begin() + mid + 1);
        vector<int> right_Label(label.begin() + mid + 1, label.end());

        merge_Sort( left, left_Label, alpha, left.size());
        merge_Sort( right, right_Label, alpha, right.size());

        Merge(pointList,label,left, right, left_Label, right_Label, alpha, left.size(), right.size());
        return;
    }



    kDTreeNode* RecbuildTree(const vector<vector<int>>& pointList, int level){
        int alpha = level % k, size = pointList.size(); 
        if(size == 0) return NULL;
        else if(size == 1) return new kDTreeNode(pointList[0]);
        else{

            vector<vector<int>> sorted_points = merge_Sort(pointList, alpha, size);
            int mid = (sorted_points.size() - 1)/ 2;      
            
            while(mid > 0 && sorted_points[mid][alpha] == sorted_points[mid - 1][alpha]) mid--;

            vector<vector<int>> Lpoints(sorted_points.begin(),  sorted_points.begin() + mid);
            vector<vector<int>> Rpoints(sorted_points.begin() + mid + 1 , sorted_points.end());  

            kDTreeNode* newNode = new kDTreeNode(sorted_points[mid]); 

            newNode->left = RecbuildTree(Lpoints, level + 1);
            newNode->right = RecbuildTree(Rpoints, level + 1);

            return newNode;
        }
    }

    kDTreeNode* RecbuildTreeLabel(const vector<vector<int>> &pointList,const vector<int> &label, int level){
        int alpha = level % k, size = pointList.size(); 
        if(size == 0) return NULL;
        else if(size == 1) return new kDTreeNode(pointList[0], label[0]);
        else{

            vector<vector<int>> sorted_points = pointList;
            vector<int> sorted_Label = label;
            this->merge_Sort(sorted_points, sorted_Label, alpha, size);

            int mid = (size - 1) / 2;
            
            while(mid > 0 && mid < size - 1 && sorted_points[mid][alpha] == sorted_points[mid - 1][alpha]) mid--;

            vector<vector<int>> Lpoints(sorted_points.begin(), sorted_points.begin() + mid);
            vector<vector<int>> Rpoints(sorted_points.begin() + mid + 1, sorted_points.end());

            vector<int> Llabel (sorted_Label.begin(), sorted_Label.begin() + mid );
            vector<int> Rlabel (sorted_Label.begin() + mid + 1, sorted_Label.end());
            kDTreeNode* newNode = new kDTreeNode(sorted_points[mid],sorted_Label[mid]);

            newNode->left = RecbuildTreeLabel(Lpoints,Llabel ,level + 1);
            newNode->right = RecbuildTreeLabel(Rpoints,Rlabel ,level + 1);

            return newNode;
        }
    }



    void buildTree(const vector<vector<int>> &pointList){
        int size = pointList.size();

        deleteTree(this->root);

        this->root = this->RecbuildTree(pointList, 0);

        this->count = size;
        return;

    }

    void buildTree(const vector<vector<int>> &pointList,const vector<int> &label){
        int size = pointList.size(), size_lable = label.size();
        if(size != size_lable) return;

        deleteTree(this->root);

        this->count = size;

        this->root = this->RecbuildTreeLabel(pointList, label, 0);

    }




    double distanceCalculate(const vector<int> a,const vector<int>b){
        double sum = 0.0;
        for(int i = 0; i < this->k; i++) sum += (a[i] - b[i])*(a[i] - b[i]); 

        return sqrt(sum);
    }


    void RecnearestNeighbour(kDTreeNode *tmp, const vector<int> &target, kDTreeNode* &best, int level, bool &full){
        if(!tmp) return;

        int alpha = level % this->k;
        double r = distanceCalculate(target, tmp->data);
        double R, current = abs(target[alpha] - tmp->data[alpha]);


        if(!tmp->left && !tmp->right){
            if(tmp->data == target){
                full = 1;
                best = tmp;
                return;
            }
            if(!best || r < distanceCalculate(target, best->data )) best = tmp;      

            return;

        }

        
        if( tmp->data[alpha] <= target[alpha]){

            if(tmp->data == target){
                full = 1;
                best = tmp;
                return;
            }
            else{
                RecnearestNeighbour(tmp->right, target, best,level + 1, full);
                if(full) return;
                if(!best ) best = tmp;
                else{
                    R = this->distanceCalculate(target, best->data );
                    r = this->distanceCalculate(target, tmp->data);
                    if (r < R) best = tmp;
                }
                R = this->distanceCalculate(target, best->data );
                if( current <= R && tmp->left ) RecnearestNeighbour(tmp->left, target, best,level + 1, full);
            }

        }else{
            RecnearestNeighbour(tmp->left, target, best,level + 1, full);
            if(full) return;
            if(!best ) best = tmp;
            else{
                R = this->distanceCalculate(target, best->data);
                r = this->distanceCalculate(target, tmp->data );
                if(r < R) best = tmp;
            }
            R = this->distanceCalculate(best->data, target);
            if(current <= R && tmp->right) RecnearestNeighbour(tmp->right, target, best,level + 1, full);

        }
        return;
    }

    void nearestNeighbour(const vector<int> &target, kDTreeNode* &best){
        bool full = 0;
        best = NULL;
        
        this->RecnearestNeighbour(this->root, target, best, 0, full);
        
    }



    
    bool compare_Nodes(const vector<int>& target, kDTreeNode* a, kDTreeNode* b){
        double Node_A = distanceCalculate(target, a->data);
        double Node_B = distanceCalculate(target, b->data ); 
        return Node_A <= Node_B;  
          
    }

    void merge_Nodes(const vector<int>& target,vector<kDTreeNode*>& nodes, vector<kDTreeNode*>& left, vector<kDTreeNode*>& right){
        
        int l = 0, r = 0;
        int l_size = left.size(), r_size = right.size();

        vector<kDTreeNode*> res;
        while(l < l_size && r < r_size){
            if(compare_Nodes(target,left[l],right[r])){
                res.push_back(left[l]);
                l++;
            }else{
                res.push_back(right[r]);
                r++;
            }
        }
        while(l < l_size){
            res.push_back(left[l]);
            l++;
        }
        while(r < r_size){
            res.push_back(right[r]);
            r++;
        }
        nodes = res;
        return;
    }

    void merge_SortNodes(const vector<int>& target, vector<kDTreeNode*>& nodes){
        
        int size = nodes.size();

        if(size <= 1) return;
        int mid = (size - 1) / 2;

        vector<kDTreeNode*> left (nodes.begin(), nodes.begin() + mid + 1);
        vector<kDTreeNode*> right (nodes.begin() + mid + 1, nodes.end());

        merge_SortNodes(target, left);
        merge_SortNodes(target, right);
        merge_Nodes(target,nodes, left,right);
    }
    void add_AllNodes(kDTreeNode* tmp, vector<kDTreeNode*>& bestList){
        if(!tmp) return;

        bestList.push_back(tmp);
        add_AllNodes(tmp->left, bestList);
        add_AllNodes(tmp->right, bestList);

    }
    void add_to_bestList(kDTreeNode* tmp, const vector<int> &target, int k ,vector<kDTreeNode *> &bestList, double &R, double r){
        int size = bestList.size();
        if(size < k){
            bestList.push_back(tmp);
            if(R < r) R = r;
            return;
        }
        int maxIdx = -1;
        for(int i = 0; i < size; i++){
                if(R <= distanceCalculate(target, bestList[i]->data)){
                    R = distanceCalculate(target, bestList[i]->data);
                    maxIdx = i;
                }
            }
        if(r < R){
            bestList.erase(bestList.begin() + maxIdx);
            bestList.push_back(tmp);
            double max = -1;
            for(int i = 0; i < size; i++){
                if(max <= distanceCalculate(target, bestList[i]->data)){
                    max = distanceCalculate(target, bestList[i]->data);
                }
            }
            R = max;
        }
        return; 
    }
    void ReckNearestNeighbour(kDTreeNode* tmp, const vector<int> &target, int k, vector<kDTreeNode *> &bestList, int level, double &R){

        if(!tmp) return;

        if(k >= this->count){
            add_AllNodes(tmp, bestList);
            return;
        }
        int alpha = level % this->k;
        double d = abs(target[alpha] - tmp->data[alpha]);
        double r = distanceCalculate(target, tmp->data);

        if(!tmp->left && !tmp->right){
            add_to_bestList(tmp, target,k, bestList, R, r); 
            return;
        }

        if(target[alpha] < tmp->data[alpha] ){
            ReckNearestNeighbour(tmp->left, target, k, bestList, level + 1,R);
            add_to_bestList(tmp, target,k, bestList, R, r);
            if(tmp->right && (d <= R || bestList.size() < k)) ReckNearestNeighbour(tmp->right, target, k, bestList, level + 1, R);  

        }else{
            ReckNearestNeighbour(tmp->right, target, k,bestList, level + 1,R);
            add_to_bestList(tmp, target,k, bestList, R,r); 
            if(tmp->left && (d <= R || bestList.size() < k)) ReckNearestNeighbour(tmp->left, target,k, bestList, level + 1,R); 
             
        } 
        return;	
    }

    void kNearestNeighbour(const vector<int> &target, int k, vector<kDTreeNode *> &bestList){
        double R = -1;

        ReckNearestNeighbour(root, target, k, bestList, 0, R);
        merge_SortNodes(target, bestList);

        return;

    }
    
    friend class kNN;
};

class kNN{

private:
    int k;
    kDTree  t;
    Dataset *X_t;
    Dataset *y_t;

public:
    kNN(int k = 5){
        this->X_t = NULL;
        this->y_t = NULL;
        this->k = k;

    }
    void fit(Dataset &X_train, Dataset &y_train){

        this->X_t = &X_train; this->y_t = &y_train;

        if(X_train.data.size()      ){
            int dim = X_train.data.front().size();

            this-> t.k = dim;

            vector<vector<int>> pointList;
            vector<int> label; 

            for(const auto &x : X_train.data){
                vector<int> node(x.begin(), x.end()); 
                pointList.push_back(node); 
            }
            for(const auto &y : y_train.data) label.push_back(y.front());

            this->t.buildTree(pointList, label);

        }

        return;
    }
    Dataset predict(Dataset &X_test){
        Dataset res;
        res.columnName.push_back("label");
        for(auto &val : X_test.data){
            vector<int> target(val.begin(), val.end() ); 
            vector<kDTreeNode *> bestList;

            this->t.kNearestNeighbour(target, this->k, bestList);

            vector<int> nums(10, 0);
            for(kDTreeNode* node : bestList) nums[node->label]++;
            
            int cnt = 0;
            for(int i = 1; i <= 9; i++) if(nums[i] > nums[cnt]) cnt = i;

            res.data.push_back({cnt});
        }
        return res;
    }
    double score(const Dataset &y_test, const Dataset &y_pred){
        int cnt = 0;

        auto ity_t = y_test.data.begin(); auto ity_p = y_pred.data.begin();  

        while(ity_t != y_test.data.end() && ity_p != y_pred.data.end()){
            if((*ity_t).front() == (*ity_p).front()) cnt++;
            ity_t++;
            ity_p++;
        } return cnt*1.0/y_test.data.size();
    }
};

// Please add more or modify as needed
